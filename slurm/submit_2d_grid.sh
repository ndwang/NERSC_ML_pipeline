#!/bin/bash
#SBATCH --job-name=vae_2d_grid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
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
# 2D GRID SEARCH
# ============================================
# Usage: sbatch slurm/submit_2d_grid.sh <param1_name> "<param1_values>" <param2_name> "<param2_values>" "<fixed_overrides>" <sweep_group>
# Example: sbatch slurm/submit_2d_grid.sh "model.latent_dim" "16 32 64 128" "training.beta" "1e-7 1e-6 1e-5 1e-4" "data=data/linear_10k.yaml training.lr=1e-3" "grid_latent_beta"
# ============================================

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <param1_name> <param1_values> <param2_name> <param2_values> <fixed_overrides> <sweep_group>" >&2
    exit 1
fi
PARAM1_NAME="$1"
IFS=' ' read -ra PARAM1_VALUES <<< "$2"
PARAM2_NAME="$3"
IFS=' ' read -ra PARAM2_VALUES <<< "$4"
FIXED_OVERRIDES="$5"
SWEEP_GROUP="$6"

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 32"

# Generate all combinations and run
run_combo() {
    local val1=$1
    local val2=$2
    local p1_short=$(echo $PARAM1_NAME | sed 's/.*\.//')
    local p2_short=$(echo $PARAM2_NAME | sed 's/.*\.//')
    local ts=$(date +%y%m%d_%H%M)
    local run_name="${p1_short}_${val1}_${p2_short}_${val2}_${ts}"
    srun $SRUN_ARGS python scripts/train.py \
        ${PARAM1_NAME}=${val1} \
        ${PARAM2_NAME}=${val2} \
        run_name=${run_name} \
        ${FIXED_OVERRIDES} \
        training.wandb.enabled=true \
        training.wandb.group=${SWEEP_GROUP} \
        > logs/${p1_short}_${val1}_${p2_short}_${val2}.log 2>&1
}
export -f run_combo
export PARAM1_NAME PARAM2_NAME FIXED_OVERRIDES SRUN_ARGS SWEEP_GROUP

parallel -j 4 --delay 0.2 run_combo ::: "${PARAM1_VALUES[@]}" ::: "${PARAM2_VALUES[@]}"

echo "Grid search complete: ${#PARAM1_VALUES[@]} x ${#PARAM2_VALUES[@]} = $((${#PARAM1_VALUES[@]} * ${#PARAM2_VALUES[@]})) configs"

# Upload all W&B logs after training completes
echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
