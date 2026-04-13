#!/bin/bash
#SBATCH --job-name=vae_1d_scan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# 1D HYPERPARAMETER SCAN
# ============================================
# Usage: sbatch slurm/submit_1d_scan.sh <param_name> <values> <fixed_overrides> <sweep_group>
# Example: sbatch slurm/submit_1d_scan.sh "model.latent_dim" "32 64 128 256" "data=data/linear_10k.yaml training.lr=1e-3" "scan_latent_dim"
# ============================================

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <param_name> <values> <fixed_overrides> <sweep_group>" >&2
    exit 1
fi
PARAM_NAME="$1"
IFS=' ' read -ra PARAM_VALUES <<< "$2"
FIXED_OVERRIDES="$3"
SWEEP_GROUP="$4"

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 32"

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
