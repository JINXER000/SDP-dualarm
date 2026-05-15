#!/bin/bash
#SBATCH -p q-hgpu-batch
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=sdp_ddpm_two_arm_threading
#SBATCH --output=/userhome/cs3/yzhchen/imitation_learning/SDP-dualarm/logs/slurm/sdp_ddpm_two_arm_threading_%j.out
#SBATCH --error=/userhome/cs3/yzhchen/imitation_learning/SDP-dualarm/logs/slurm/sdp_ddpm_two_arm_threading_%j.err
# Uncomment and set if your cluster requires an explicit address for mail-type=ALL:
# #SBATCH --mail-user=your.email@example.com

# Optional array / seed pattern (uncomment to use):
# SEEDS=(0 1 2)
# SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail

# Hydra override for sdp_ddpm_5layer_dmg.yaml `n_demo` (default matches yaml).
# Usage: sbatch .../job_sdp_ddpm_two_arm_threading.sh 100   # first arg overrides
N_DEMO="${1:-200}"

LOGDIR=/userhome/cs3/yzhchen/imitation_learning/SDP-dualarm/logs/slurm
mkdir -p "${LOGDIR}"

source /userhome/cs3/yzhchen/miniconda3/etc/profile.d/conda.sh
conda activate sdp

cd /userhome/cs3/yzhchen/imitation_learning/SDP-dualarm

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=sdp_ddpm_5layer_dmg \
  task_name=two_arm_threading \
  n_demo="${N_DEMO}"
