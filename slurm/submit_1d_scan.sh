#!/bin/bash
#SBATCH --job-name=vae_1d_scan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# 1D HYPERPARAMETER SCAN
# ============================================
# Usage: sbatch submit_1d_scan.sh
# Runs multiple configs in parallel using GNU parallel
# ============================================

# --- CONFIGURATION ---
PARAM_NAME="model.latent_dim"           # Parameter to scan (dot notation)
PARAM_VALUES=(32 64 128)                # Values to try
FIXED_OVERRIDES="data=data/sectioned_10k.yaml training.lr=1e-3 training.beta=1e-5 training.gamma=1e-4 training.delta=1e-4 training.batch_size=512"
SWEEP_GROUP="scan_latent_dim"

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 16"

# Generate run commands
run_single() {
    local val=$1
    local param_short=$(echo $PARAM_NAME | sed 's/.*\.//')
    local ts=$(date +%y%m%d_%H%M)
    local run_name="${param_short}_${val}_${ts}"
    srun $SRUN_ARGS python scripts/train.py \
        ${PARAM_NAME}=${val} \
        run_name=${run_name} \
        ${FIXED_OVERRIDES} \
        training.wandb.enabled=true \
        training.wandb.group=${SWEEP_GROUP} \
        > logs/${param_short}_${val}.log 2>&1
}
export -f run_single
export PARAM_NAME FIXED_OVERRIDES SRUN_ARGS SWEEP_GROUP

parallel -j 4 --delay 0.2 run_single ::: "${PARAM_VALUES[@]}"

# Upload all W&B logs after training completes
echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
