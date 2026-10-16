#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""[wavlm_voc1_1k] Used here with --s3prl-upstream wavlm_large --layer 21 and the
WavLM k-means km_1000.mdl (see run.sh); WavLM-large also takes 16kHz input with a
320-sample stride, so the HUBERT_* constants below still hold. Original docstring:

Extract HuBERT k-means tokens and dump them as ParallelWaveGAN training
features.

Adapted from ../../phon_voc1/local/preprocess_phon.py, with the content
tokenizer swapped for the one used by the VoxtLM recipe (espnet
egs2/voxtlm_v1, kmeans_feature="hubert_base/6"): s3prl "hubert_base" layer-6
features, labelled with a pre-trained k-means model (km_1000.mdl). Feature
extraction goes through the same espnet2 S3prlFrontend, and the labelling is
the same nearest-centroid computation as ApplyKmeans in espnet's
egs2/TEMPLATE/asr1/pyscripts/feats/dump_km_label.py, so the tokens here match
the ones the speech LM was trained on.

Unlike the vctk/hubert_voc1 recipe, which reads precomputed tokens from a
kaldi-style text file, tokens are computed here on the exact 16kHz waveform
that is dumped as the training target, so the two are always frame-aligned.
LJSpeech is single-speaker, so "feats" is just the token column, shape (T, 1).
"""

import argparse
import logging
import os

import joblib
import librosa
import numpy as np
import resampy
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5

# HuBERT-base consumes 16kHz audio with a 320-sample (20ms) frame stride.
HUBERT_SAMPLING_RATE = 16000
HUBERT_HOP_SIZE = 320


class ApplyKmeans(object):
    """Nearest-centroid labelling, same as espnet's dump_km_label.py."""

    def __init__(self, km_path):
        km_model = joblib.load(km_path)
        self.C_np = km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

    @property
    def num_clusters(self):
        return self.C_np.shape[1]

    @property
    def feat_dim(self):
        return self.C_np.shape[0]

    def __call__(self, x):
        dist = (x**2).sum(1, keepdims=True) - 2 * np.matmul(x, self.C_np) + self.Cnorm_np
        return np.argmin(dist, axis=1)


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and extract HuBERT k-means token features."
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including wav files. you need to specify either scp or rootdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--km-path",
        type=str,
        required=True,
        help="pre-trained k-means model (joblib-dumped sklearn model, e.g. km_1000.mdl).",
    )
    parser.add_argument(
        "--s3prl-upstream",
        type=str,
        default="hubert_base",
        help="s3prl upstream name used to extract the features clustered by --km-path.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="upstream layer used to extract the features clustered by --km-path.",
    )
    parser.add_argument(
        "--s3prl-download-dir",
        type=str,
        default=None,
        help="directory to cache the s3prl upstream checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device used to run the upstream model.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # the dumped waveform is also the upstream input, so it has to be at the
    # upstream's rate, and one token must cover exactly hop_size samples
    assert config["sampling_rate"] == HUBERT_SAMPLING_RATE, (
        f"sampling_rate must be {HUBERT_SAMPLING_RATE} "
        f"(got {config['sampling_rate']})."
    )
    assert config["hop_size"] == HUBERT_HOP_SIZE, (
        f"hop_size must be {HUBERT_HOP_SIZE} (got {config['hop_size']})."
    )

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # load the upstream model and the k-means model
    from espnet2.asr.frontend.s3prl import S3prlFrontend  # NOQA

    frontend = S3prlFrontend(
        fs=HUBERT_SAMPLING_RATE,
        frontend_conf={"upstream": args.s3prl_upstream},
        download_dir=args.s3prl_download_dir,
        multilayer_feature=False,
        layer=args.layer,
    )
    frontend = frontend.to(args.device).eval()
    apply_kmeans = ApplyKmeans(args.km_path)
    assert apply_kmeans.feat_dim == frontend.output_size(), (
        f"{args.km_path} was fit on {apply_kmeans.feat_dim}-dim features, but "
        f"{args.s3prl_upstream} layer {args.layer} gives {frontend.output_size()}-dim."
    )
    assert (
        config["generator_params"]["num_embs"] == apply_kmeans.num_clusters
    ), (
        f"generator_params.num_embs ({config['generator_params']['num_embs']}) != "
        f"number of clusters in {args.km_path} ({apply_kmeans.num_clusters})."
    )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # resume support: skip utterances already dumped by a previous
        # (possibly interrupted) run
        if config["format"] == "hdf5":
            out_path = os.path.join(args.dumpdir, f"{utt_id}.h5")
        else:
            out_path = os.path.join(args.dumpdir, f"{utt_id}-feats.npy")
        if os.path.exists(out_path):
            continue

        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # downsample to 16kHz: both the vocoder target and the upstream input
        if fs != config["sampling_rate"]:
            audio = resampy.resample(audio, fs, config["sampling_rate"], axis=0)

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        # extract layer features and label them: (T,)
        with torch.no_grad():
            x = torch.from_numpy(audio).float().view(1, -1).to(args.device)
            x_lens = torch.tensor([x.shape[1]], dtype=torch.long)
            hs, hs_lens = frontend(x, x_lens)
        hs = hs[0, : hs_lens[0]].cpu().numpy()
        tokens = apply_kmeans(hs)

        feats = tokens.reshape(-1, 1).astype(np.float32)

        # make sure the audio length and feature length are matched
        logging.info(f"Mod: {len(audio) - len(feats) * config['hop_size']}")
        feats = feats[: len(audio) // config["hop_size"]]
        audio = audio[: len(feats) * config["hop_size"]]
        assert len(feats) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio = audio * config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warning(
                f"{utt_id} causes clipping. "
                "it is better to re-consider global gain scale."
            )
            continue

        # save
        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "feats",
                feats.astype(np.float32),
            )
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                feats.astype(np.float32),
                allow_pickle=False,
            )
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()
