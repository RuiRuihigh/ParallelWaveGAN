#!/usr/bin/env bash
#SBATCH --job-name=ljs_hubert_k1000_tts
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk1000_ttskmeans/log/myrun.%j.out
#SBATCH --error=/export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk1000_ttskmeans/log/myrun.%j.err

# Controller job for the CLSP grid: it stays on the cpu partition and blocks
# while run.sh submits the real work through utils/slurm.pl. Default for GPU
# jobs is conf/slurm.conf (-p gpu, 1080 Ti); training and VERSA override it to
# the A100 nodes via CUDA_SLURM_CONF.
#   stage 1 (HuBERT+kmeans, CPU)-> train_cmd, conf/slurm.conf, -p cpu
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

cd /export/fs06/mliang17/phonologicaltokenizer_modify/ParallelWaveGAN/egs/ljspeech/hubert_voc_v1Balk1000_ttskmeans

. ./path.sh
. ./cmd.sh

# Full run from scratch: the k-means model differs from ../hubert_voc1
# (lm1_k1000_Bal_ttskmeans instead of lm1_k1000_Bal), so tokens must be
# re-extracted and the vocoder retrained. LJSpeech and the s3prl checkpoint
# are symlinked from ../phon_voc1 and ../hubert_voc1, so stage -1 skips the
# download.

# Stage -1 ~ 1: download, data prep, HuBERT k-means tokenization (CPU only).
./run.sh --stage -1 --stop_stage 1

# Stage 2: training on an A100.
# NOTE: train_max_steps is set to 400k in conf/hifigan_hubert.v1.yaml, so
# training ends by itself and stages 3/4 run automatically -- no manual stop.
CUDA_SLURM_CONF=conf/slurm_clsp_a100.conf ./run.sh --stage 2 --stop_stage 2

# Checkpoint that stages 3/4 score: train_max_steps is 400k, so this is the
# final one, and it matches the step count ../phon_voc1, ../hubert_voc1 and
# ../hubert_voc_v1Balk2000 were scored at.
checkpoint=exp/train_nodev_ljspeech_hifigan_hubert.v1/checkpoint-400000steps.pkl

# Stage 3: decoding on the regular gpu partition.
./run.sh --stage 3 --stop_stage 3 \
    --checkpoint "${checkpoint}" \
    --decode_sets eval

# Stage 4: VERSA objective evaluation on the A100 nodes. Loading every metric
# in conf/versa.yaml at once peaks at ~11.5GB reserved, which does not fit a
# 1080 Ti's 10.9GB: on the regular gpu partition every job fails to load
# whisper_wer with CUDA OOM and silently scores without it (seen in phon_voc1).
CUDA_SLURM_CONF=conf/slurm_clsp_a100.conf ./run.sh --stage 4 --stop_stage 4 \
    --checkpoint "${checkpoint}" \
    --decode_sets eval \
    --versa_nj 1
