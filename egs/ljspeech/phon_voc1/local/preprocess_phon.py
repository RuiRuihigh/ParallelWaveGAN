#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract Sony Phonological-Tokenizer tokens and dump them as ParallelWaveGAN
training features.

Adapted from ../../vctk/phon_voc1/local/preprocess_phon.py: LJSpeech is a
single-speaker corpus, so there is no need for an x-vector (or any other)
speaker condition. The dumped "feats" array is just the token column, shape
(T, 1), and is consumed by DiscreteSymbolHiFiGANGenerator (num_spk_embs=0)
via the generic aux-feature Collater in parallel_wavegan/bin/train.py.
"""

import argparse
import logging
import os
import sys

import librosa
import numpy as np
import resampy
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset, AudioSCPDataset
from parallel_wavegan.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and extract Phonological-Tokenizer token features."
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
        "--tokenizer-dir",
        type=str,
        required=True,
        help=(
            "directory containing the downloaded Sony Phonological-Tokenizer "
            "files (tokenizer.py, ssl.pth, centroids.npy)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device used to run the tokenizer.",
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

    # load the phonological tokenizer
    sys.path.insert(0, args.tokenizer_dir)
    from tokenizer import PhonologicalTokenizer  # NOQA

    tokenizer = PhonologicalTokenizer(
        ssl_model_path=os.path.join(args.tokenizer_dir, "ssl.pth"),
        centroids_path=os.path.join(args.tokenizer_dir, "centroids.npy"),
        device=args.device,
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

        # downsample to the target output sampling rate (the waveform the
        # vocoder is trained to reconstruct)
        if fs != config["sampling_rate"]:
            audio_out = resampy.resample(audio, fs, config["sampling_rate"], axis=0)
        else:
            audio_out = audio

        # trim silence
        if config["trim_silence"]:
            audio_out, _ = librosa.effects.trim(
                audio_out,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        # tokenizer expects a 16kHz wav file on disk
        wav_16k = audio if fs == 16000 else resampy.resample(audio, fs, 16000, axis=0)
        tmp_wav_path = os.path.join(args.dumpdir, f".{utt_id}_16k.wav")
        sf.write(tmp_wav_path, wav_16k, 16000)

        # extract discrete phonological tokens: (T,)
        with torch.no_grad():
            tokens = tokenizer(tmp_wav_path)[0].cpu().numpy()

        os.remove(tmp_wav_path)

        feats = tokens.reshape(-1, 1).astype(np.float32)

        # make sure the audio length and feature length are matched
        logging.info(f"Mod: {len(audio_out) - len(feats) * config['hop_size']}")
        feats = feats[: len(audio_out) // config["hop_size"]]
        audio_out = audio_out[: len(feats) * config["hop_size"]]
        assert len(feats) * config["hop_size"] == len(audio_out)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio_out = audio_out * config["global_gain_scale"]
        if np.abs(audio_out).max() >= 1.0:
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
                audio_out.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "feats",
                feats.astype(np.float32),
            )
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio_out.astype(np.float32),
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
