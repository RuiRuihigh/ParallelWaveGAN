#!/usr/bin/env bash
#SBATCH --job-name=ljs_wavlm_k1000_rs
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/wavlm_voc1_1k/log/myrun_resume.%j.out
#SBATCH --error=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/wavlm_voc1_1k/log/myrun_resume.%j.err

# Controller job for the CLSP grid: it stays on the cpu partition and blocks
# while run.sh submits the real work through utils/slurm.pl. Default for GPU
# jobs is conf/slurm.conf (-p gpu, 1080 Ti); training and VERSA override it to
# the A100 nodes via CUDA_SLURM_CONF.
#   stage 1 (WavLM+kmeans, CPU) -> train_cmd, conf/slurm.conf, -p cpu
#   stage 2 (training)          -> cuda_cmd, conf/slurm_clsp_a100.conf
#   stage 3 (decoding)          -> cuda_cmd, conf/slurm.conf, -p gpu
#   stage 4 (VERSA)             -> cuda_cmd, conf/slurm_clsp_a100.conf
# Stage 4's model pre-fetch runs inline on this controller, hence 8x1900M RAM.
# Submit with: sbatch myrun.sh

# conda.sh is missing on some nodes (e.g. c17), so put the env on PATH directly
# instead of `conda activate parallelwavegan`; sub-jobs inherit it (--export=ALL).
export CONDA_PREFIX=/home/mliang17/.conda/envs/parallelwavegan
export PATH="${CONDA_PREFIX}/bin:${PATH}"

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/wavlm_voc1_1k

. ./path.sh
. ./cmd.sh

# 2026-10-11 resume. The first run (1819693 / A100 job 1819893) was cancelled
# at ~150k steps to free an A100 for the LM training (QOS gpua100: at most 3
# A100s per user). Stages -1..1 (tokens in dump/) are done and are not rerun.
# Training resumes from checkpoint-150000steps.pkl with a 24h A100 limit;
# ~250k steps left at ~5.5 it/s is ~12.5h. If the 24h limit is hit, the
# whole job ends there (no automatic resubmission).
# slurm.pl overwrites train.log, so the first run's log is kept as
# train.log.until150k.
expdir=exp/train_nodev_ljspeech_hifigan_wavlm.v1
[ -e "${expdir}/train.log.until150k" ] || cp -p "${expdir}/train.log" "${expdir}/train.log.until150k"

# Stage 2: training on an A100, 24h limit. CUDA_SLURM_CONF is expanded
# unquoted into cuda_cmd by cmd.sh, so the extra slurm.pl option rides along.
CUDA_SLURM_CONF="conf/slurm_clsp_a100.conf --time 24:00:00" ./run.sh --stage 2 --stop_stage 2 \
    --resume "${expdir}/checkpoint-150000steps.pkl"

# Checkpoint that stages 3/4 score: train_max_steps is 400k, so this is the
# final one, and it matches the step count ../phon_voc1, ../hubert_voc1 and
# ../hubert_voc_v1Balk2000 were scored at.
checkpoint=exp/train_nodev_ljspeech_hifigan_wavlm.v1/checkpoint-400000steps.pkl

# Stage 3: decoding on the regular gpu partition.
./run.sh --stage 3 --stop_stage 3 \
    --checkpoint "${checkpoint}" \
    --decode_sets eval

# Stage 4: VERSA objective evaluation on the A100 nodes (4h limit, so backfill can place it). Loading every metric
# in conf/versa.yaml at once peaks at ~11.5GB reserved, which does not fit a
# 1080 Ti's 10.9GB: on the regular gpu partition every job fails to load
# whisper_wer with CUDA OOM and silently scores without it (seen in phon_voc1).
CUDA_SLURM_CONF="conf/slurm_clsp_a100.conf --time 04:00:00" ./run.sh --stage 4 --stop_stage 4 \
    --checkpoint "${checkpoint}" \
    --decode_sets eval \
    --versa_nj 1
