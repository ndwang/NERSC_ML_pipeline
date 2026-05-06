#!/bin/bash
#SBATCH --job-name=vae_scan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu
#SBATCH --time=08:00:00

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
ml load parallel

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 32"

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

parallel -j "$SLURM_NTASKS" --delay 0.2 run_single ::: "${PARAM_VALUES[@]}"

echo "Scan complete: ${#PARAM_VALUES[@]} configs"

echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
