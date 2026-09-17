#!/usr/bin/env bash
#SBATCH --job-name=stages-vocoder-ljspeech
#SBATCH --partition=RM-shared
#SBATCH --account=cis210027p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=1-00:00:00
#SBATCH --output=/ocean/projects/cis210027p/mliang4/ParallelWaveGAN/egs/ljspeech/phon_voc1/log/myrun.%j.out
#SBATCH --error=/ocean/projects/cis210027p/mliang4/ParallelWaveGAN/egs/ljspeech/phon_voc1/log/myrun.%j.err

source /ocean/projects/cis210027p/mliang4/miniconda3/etc/profile.d/conda.sh
conda activate parallelwavegan

set -e
set -u
set -o pipefail

cd /ocean/projects/cis210027p/mliang4/ParallelWaveGAN/egs/ljspeech/phon_voc1

. ./path.sh
. ./cmd.sh

#--partition=GPU-shared
#--gres=gpu:v100-32:1

# ./run.sh --stage -1 --stop_stage 1
# ./run.sh --stage 2 --stop_stage 2
# ./run.sh --stage 3 --stop_stage 3 \
#     --checkpoint exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-50000steps.pkl \
#     --decode_sets eval
./run.sh --stage 4 --stop_stage 4 \
    --checkpoint exp/train_nodev_ljspeech_hifigan_phon.v1/checkpoint-50000steps.pkl \
    --decode_sets eval \
    --versa_nj 2
