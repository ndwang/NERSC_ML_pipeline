#!/bin/bash
#SBATCH --job-name=vae_single
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# SINGLE TRAINING RUN
# ============================================
# Usage: sbatch slurm/submit_single.sh <run_prefix> <sweep_group> <overrides>
# Example: sbatch slurm/submit_single.sh "latent128" "scan_latent" "data=data/sectioned_10k.yaml model.latent_dim=128 training.lr=1e-3"
# ============================================

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <run_prefix> <sweep_group> <overrides>" >&2
    exit 1
fi
RUN_PREFIX="$1"
SWEEP_GROUP="$2"
OVERRIDES="$3"

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

# Generate unique run name with timestamp
RUN_NAME="${RUN_PREFIX}_$(date +%y%m%d_%H%M)"

# Run training with W&B in offline mode
python scripts/train.py $OVERRIDES run_name=${RUN_NAME} training.wandb.enabled=true training.wandb.group=${SWEEP_GROUP}

# Upload W&B logs after training completes
echo "Syncing W&B logs..."
wandb sync runs/${RUN_NAME}/wandb/offline-run-* --sync-all
