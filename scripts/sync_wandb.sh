#!/bin/bash
# Sync all offline wandb runs

for run_dir in wandb/offline-run-*; do
    [ -d "$run_dir" ] || continue
    echo "Syncing $run_dir ..."
    wandb sync "$run_dir"
done
