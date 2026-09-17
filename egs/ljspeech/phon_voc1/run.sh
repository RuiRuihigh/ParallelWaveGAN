#!/bin/bash

# Adapted from ../voc1/run.sh (data download/prep stages) and
# ../../vctk/phon_voc1/run.sh (feature extraction/training stages): content
# token comes from Sony/Phonological-Tokenizer (called directly on each wav)
# instead of a mel-spectrogram. LJSpeech is single-speaker, so unlike
# ../../vctk/phon_voc1 there is no speaker condition (no x-vector extraction,
# see local/preprocess_phon.py and conf/hifigan_phon.v1.yaml).

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan_phon.v1.yaml

# directory path setting
download_dir=downloads # directory to save downloaded files
dumpdir=dump           # directory to dump features

# Sony Phonological-Tokenizer settings
tokenizer_dir=downloads/phonological_tokenizer # dir with tokenizer.py, ssl.pth, centroids.npy

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)
decode_sets="dev eval" # space-separated list of sets to decode (stage 3)
                        # and score (stage 4) (e.g. "eval" to skip dev)

# VERSA objective evaluation settings (stage 4)
versa_config=conf/versa.yaml
versa_cache=downloads/versa_cache   # symlinked to ../../vctk/phon_voc1's
                                     # cache to reuse already-downloaded
                                     # metric models instead of re-fetching
versa_nj=8

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_nodev" # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="eval"         # name of evaluation data direcotry

set -euo pipefail

if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
    echo "Stage -1: Data download"
    local/data_download.sh "${download_dir}"
fi

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    local/data_prep.sh \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        "${download_dir}/LJSpeech-1.1" data
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    if [ ! -e "${tokenizer_dir}/ssl.pth" ] || [ ! -e "${tokenizer_dir}/centroids.npy" ]; then
        echo "Valid --tokenizer_dir is not provided."
        echo "Download it first, e.g.:"
        cat << EOF
python -c "from huggingface_hub import snapshot_download; \\
    snapshot_download(repo_id='Sony/Phonological-Tokenizer', local_dir='${tokenizer_dir}')"
EOF
        exit 1
    fi
    # extract raw features (no normalization stage: tokens and waveform
    # samples are used as-is, unlike the mel-spectrogram pipeline in
    # ../voc1/run.sh)
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        # --num-threads 4: on RM-shared, memory scales with cpus-per-task
        # (DefMemPerCPU=1900M), so this also buys ~7.6GB per job instead of
        # the 1.9GB/1-core default, which WavLM-large needs to avoid OOM
        # (see ../../vctk/phon_voc1/run.sh).
        ${train_cmd} --num-threads 4 JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            local/preprocess_phon.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --tokenizer-dir "${tokenizer_dir}" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ -z "${tag}" ]; then
    expdir="exp/${train_set}_ljspeech_$(basename "${conf}" .yaml)"
else
    expdir="exp/${train_set}_ljspeech_${tag}"
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in ${decode_sets}; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Objective evaluation (VERSA)"
    if ! python -c "import versa" > /dev/null 2>&1; then
        echo "VERSA is not installed. Install it first, e.g.:"
        echo "  pip install git+https://github.com/wavlab-speech/versa.git#egg=versa-speech-audio-toolkit --no-build-isolation"
        exit 1
    fi
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    for name in ${decode_sets}; do
        _gen_dir="${outdir}/${name}"
        _eval_dir="${_gen_dir}/scoring/versa_eval"
        mkdir -p "${_eval_dir}"

        # decode.py writes <utt_id>_gen.wav directly (no scp file), so build
        # one ourselves to split the same way as a normal kaldi wav.scp.
        # NOTE: strip the "_gen" suffix too, not just ".wav" -- otherwise the
        # keys here ("<utt_id>_gen") won't match data/${name}/{wav.scp,text}
        # ("<utt_id>") and every utterance gets silently skipped as "Ground
        # truth not found".
        find "${_gen_dir}" -maxdepth 1 -iname "*.wav" | sort \
            | awk -F/ '{fn=$NF; sub(/\.wav$/, "", fn); sub(/_gen$/, "", fn); print fn, $0}' \
            > "${_eval_dir}/pred.scp"
        _n_pred=$(wc -l < "${_eval_dir}/pred.scp")
        echo "Scoring ${_n_pred} generated utterances for ${name}."

        # --gt/--text are looked up by utt_id key internally (not iterated
        # in lockstep with --pred), so the *full* data/${name}/{wav.scp,text}
        # can be passed as-is -- no need to filter down to pred's utt_ids.
        _gt_wavscp="data/${name}/wav.scp"
        _gt_text="data/${name}/text"

        # pre-fetch all metric models once, sequentially, before the
        # parallel jobs below -- avoids a FileExistsError race if multiple
        # workers try to populate the shared versa_cache concurrently on
        # first use.
        # NOTE: this runs inline on whatever node is executing run.sh itself
        # (the myrun.sh outer job, which stays on RM-shared with no GPU), so
        # no --use_gpu here -- we only need the download+cache side effect,
        # not real inference. The actual scoring below is what runs on GPU.
        head -n 2 "${_eval_dir}/pred.scp" > "${_eval_dir}/warmup_pred.scp"
        echo "Pre-fetching VERSA metric models..."
        python -m versa.bin.scorer \
            --pred "${_eval_dir}/warmup_pred.scp" \
            --gt "${_gt_wavscp}" \
            --text "${_gt_text}" \
            --score_config "${versa_config}" \
            --cache_folder "${versa_cache}" \
            --output_file "${_eval_dir}/warmup_result.txt" \
            --io soundfile \
            --scoring_mode utterance

        _nj=$(( versa_nj < _n_pred ? versa_nj : _n_pred ))
        _split_pred=""
        for n in $(seq "${_nj}"); do
            _split_pred+="${_eval_dir}/pred.${n} "
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${_eval_dir}/pred.scp" ${_split_pred}

        ${cuda_cmd} --gpu 1 JOB=1:"${_nj}" "${_eval_dir}/versa_eval.JOB.log" \
            python -m versa.bin.scorer \
                --pred "${_eval_dir}/pred.JOB" \
                --gt "${_gt_wavscp}" \
                --text "${_gt_text}" \
                --score_config "${versa_config}" \
                --cache_folder "${versa_cache}" \
                --output_file "${_eval_dir}/result.JOB.txt" \
                --io soundfile \
                --scoring_mode utterance \
                --resume \
                --use_gpu

        python local/aggregate_versa_eval.py \
            --logdir "${_eval_dir}" \
            --scoredir "${_eval_dir}" \
            --nj "${_nj}"
        echo "VERSA scoring for ${name} done. See ${_eval_dir}/avg_result.txt"
    done
    echo "Successfully finished objective evaluation."
fi
echo "Finished."
