#!/usr/bin/env bash
#SBATCH --job-name=ljs_phon_voc1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1/log/myrun.%j.err

# Controller job for the CLSP grid: it stays on the cpu partition and blocks
# while run.sh submits the real work through utils/slurm.pl. Default for GPU
# jobs is conf/slurm.conf (-p gpu, 1080 Ti); training and VERSA override it to
# the A100 nodes via CUDA_SLURM_CONF.
#   stage 1 (tokenization, CPU) -> train_cmd, conf/slurm.conf, -p cpu
#   stage 2 (training)          -> cuda_cmd, conf/slurm_clsp_a100.conf
#   stage 3 (decoding)          -> cuda_cmd, conf/slurm.conf, -p gpu
#   stage 4 (VERSA)             -> cuda_cmd, conf/slurm_clsp_a100.conf
# Stage 4's model pre-fetch runs inline on this controller, hence 8x1900M RAM.
# Submit with: sbatch myrun.sh

source /opt/slurm_files/software/conda25/2025-08-21/etc/profile.d/conda.sh
conda activate parallelwavegan

set -e
set -u
set -o pipefail

cd /export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/phon_voc1

. ./path.sh
. ./cmd.sh

# Stages -1 ~ 2 are commented out: data/features are already dumped, and
# training keeps running in the original controller job (1799568). This
# submission only decodes and scores an intermediate checkpoint.
checkpoint=exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-400000steps.pkl

# # Stage -1 ~ 1: download, data prep, tokenization (CPU only).
# ./run.sh --stage -1 --stop_stage 1

# # Stage 2: training on an A100.
# # NOTE: train_max_steps in conf/hifigan_phon.v1.yaml is 2.5M; pass
# # --stop_stage / --resume to split the run as needed.
# CUDA_SLURM_CONF=conf/slurm_clsp_a100.conf ./run.sh --stage 2 --stop_stage 2

# # Stage 3: decoding on the regular gpu partition.
# # (already done for 400k by job 1800136)
# ./run.sh --stage 3 --stop_stage 3 \
#     --checkpoint "${checkpoint}" \
#     --decode_sets eval

# Stage 4: VERSA objective evaluation on the A100 nodes. Loading every metric
# in conf/versa.yaml at once peaks at ~11.5GB reserved, which does not fit a
# 1080 Ti's 10.9GB: on the regular gpu partition (job 1800136) all 16 jobs
# failed to load whisper_wer with CUDA OOM and silently scored without it.
CUDA_SLURM_CONF=conf/slurm_clsp_a100.conf ./run.sh --stage 4 --stop_stage 4 \
    --checkpoint "${checkpoint}" \
    --decode_sets eval \
    --versa_nj 1
