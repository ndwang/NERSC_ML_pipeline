#!/bin/bash
# 1D scan submission — auto-allocates ceil(n_values / 4) nodes so all runs are parallel.
#
# Usage: bash slurm/submit_1d_scan.sh [--time HH:MM:SS] \
#          <param_name> "<values>" "<fixed_overrides>" <sweep_group>
#
# Example (6 runs → 2 nodes):
#   bash slurm/submit_1d_scan.sh \
#     "model.latent_dim" "16 32 64 128 256 512" \
#     "training.lr=1e-3" "v2_scan_latent_dim"

TIME="08:00:00"
if [[ "$1" == "--time" ]]; then
    TIME="$2"; shift 2
fi

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 [--time HH:MM:SS] <param_name> <values> <fixed_overrides> <sweep_group>" >&2
    exit 1
fi

IFS=' ' read -ra VALS <<< "$2"
N=${#VALS[@]}
NODES=$(( (N + 3) / 4 ))

echo "Submitting ${N} runs across ${NODES} node(s), time limit ${TIME}."
sbatch --nodes="$NODES" --time="$TIME" "$(dirname "$0")/scan_worker.sh" "$@"
