#!/bin/bash
# 2D grid submission — auto-allocates ceil(n_combos / 4) nodes so all runs are parallel.
#
# Usage: bash slurm/submit_grid.sh [--time HH:MM:SS] \
#          <param1_name> "<param1_values>" \
#          <param2_name> "<param2_values>" \
#          "<fixed_overrides>" <sweep_group>
#
# Example (8 runs → 2 nodes):
#   bash slurm/submit_grid.sh \
#     "model.latent_dim" "128 256" \
#     "training.beta" "2e-5 5e-5 1e-4 1e-3" \
#     "training.lr=1e-3" "v2_grid_latent_beta"

TIME="08:00:00"
if [[ "$1" == "--time" ]]; then
    TIME="$2"; shift 2
fi

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 [--time HH:MM:SS] <param1_name> <param1_values> <param2_name> <param2_values> <fixed_overrides> <sweep_group>" >&2
    exit 1
fi

IFS=' ' read -ra P1 <<< "$2"
IFS=' ' read -ra P2 <<< "$4"
N=$(( ${#P1[@]} * ${#P2[@]} ))
NODES=$(( (N + 3) / 4 ))

echo "Submitting ${N} runs (${#P1[@]} × ${#P2[@]}) across ${NODES} node(s), time limit ${TIME}."
sbatch --nodes="$NODES" --time="$TIME" "$(dirname "$0")/grid_worker.sh" "$@"
