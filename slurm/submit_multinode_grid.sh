#!/bin/bash
#SBATCH --job-name=vae_grid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# MULTI-NODE 2D GRID SEARCH
# ============================================
# Usage: sbatch [--nodes=N] slurm/submit_multinode_grid.sh \
#          <param1_name> "<param1_values>" \
#          <param2_name> "<param2_values>" \
#          "<fixed_overrides>" <sweep_group>
#
# --nodes=N controls parallelism: N nodes × 4 GPUs each.
# Total runs = |param1| × |param2|; size the node count accordingly.
#
# Example (8 runs, 2 nodes):
#   sbatch --nodes=2 slurm/submit_multinode_grid.sh \
#     "model.latent_dim" "128 256" \
#     "training.beta" "2e-6 5e-6 2e-5 5e-5" \
#     "training.lr=1e-3" "v2_scan5_beta_fine"
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

parallel -j $SLURM_NTASKS --delay 0.2 run_combo ::: "${PARAM1_VALUES[@]}" ::: "${PARAM2_VALUES[@]}"

echo "Grid search complete: ${#PARAM1_VALUES[@]} x ${#PARAM2_VALUES[@]} = $((${#PARAM1_VALUES[@]} * ${#PARAM2_VALUES[@]})) configs"

echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
