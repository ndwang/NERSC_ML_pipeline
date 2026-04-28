#!/bin/bash
# Sync all offline wandb runs that haven't been synced yet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

while IFS= read -r run_dir; do
    if [ -z "$(find "$run_dir" -maxdepth 1 -name "*.wandb.synced" 2>/dev/null)" ]; then
        echo "Syncing $run_dir ..."
        wandb sync "$run_dir"
    fi
done < <(find "$PROJECT_ROOT/runs" -maxdepth 3 -type d -name "offline-run-*")
