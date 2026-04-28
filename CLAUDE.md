# VAE Training Pipeline

Variational Autoencoder training pipeline for frequency map data on NERSC Perlmutter.

## Project Structure

```
├── configs/           # YAML configuration files
│   ├── model/        # Model architectures (vae2d, residual_vae2d)
│   ├── training/     # Training hyperparameters
│   └── data/         # Dataset paths
├── scripts/          # Entry point scripts
│   └── train.py      # Main training script
├── slurm/            # NERSC job submission scripts
└── beam_vae/         # Installable package (pip install -e .)
    ├── models/       # VAE2D, ResidualVAE2D
    ├── data/         # FrequencyMapDataset
    ├── training/     # Trainer, losses
    └── utils/        # Config loading, activations
```

## Quick Commands

```bash
# Train with defaults
python scripts/train.py

# Override hyperparameters
python scripts/train.py model.latent_dim=128 training.beta=1e-5

# Use residual model
python scripts/train.py model=model/residual_vae2d.yaml

# Resume from checkpoint
python scripts/train.py --resume runs/my_run/vae_best.pth

# Submit to SLURM (run from project root)
sbatch slurm/submit_single.sh <run_prefix> <sweep_group> "<overrides>"
sbatch slurm/submit_single.sh "latent128" "scan_latent" "data=data/sectioned_10k.yaml model.latent_dim=128 training.lr=1e-3"

sbatch slurm/submit_1d_scan.sh <param_name> "<values>" "<fixed_overrides>" <sweep_group>
sbatch slurm/submit_1d_scan.sh "model.latent_dim" "32 64 128 256" "data=data/linear_10k.yaml training.lr=1e-3" "scan_latent_dim"

sbatch slurm/submit_2d_grid.sh <param1_name> "<param1_values>" <param2_name> "<param2_values>" "<fixed_overrides>" <sweep_group>
sbatch slurm/submit_2d_grid.sh "model.latent_dim" "16 32 64 128" "training.beta" "1e-7 1e-6 1e-5 1e-4" "data=data/linear_10k.yaml training.lr=1e-3" "grid_latent_beta"
```

## Post-Training Analysis

Two scripts for analyzing completed runs:

```bash
# 1. Loss curve analysis (fast, no GPU needed)
#    Summary table, convergence speed, overfitting check, loss trajectory
python scripts/analyze_losses.py runs/beta_*_260401_*           # Compare sweep
python scripts/analyze_losses.py runs/beta_1e-5_260401_1523     # Single run
python scripts/analyze_losses.py runs/lr_*_260401_* --convergence
python scripts/analyze_losses.py runs/lr_*_260401_* --overfitting
python scripts/analyze_losses.py runs/beta_*_260401_* --all     # Everything

# 2. Model output analysis (runs inference, slower)
#    Reconstructions, latent space PCA, scale/centroid R², dimension utilization
python scripts/analyze_model.py runs/beta_1e-5_260401_1523
python scripts/analyze_model.py runs/beta_1e-5_260401_1523 --only recon latent
python scripts/analyze_model.py runs/beta_1e-5_260401_1523 --only scales centroids dims
```

Workflow: run `analyze_losses.py` first to identify interesting runs, then `analyze_model.py` on the winners for detailed inspection.

## Key Files

- `beam_vae/models/vae2d.py` - Standard VAE architecture
- `beam_vae/models/residual_vae2d.py` - VAE with residual blocks
- `beam_vae/training/trainer.py` - Training loop, checkpointing, and resume
- `beam_vae/utils/config.py` - YAML config loading with CLI overrides
- `beam_vae/utils/validation.py` - Pydantic config schema validation

## Data

- Input: 15-channel 64x64 frequency maps stored as `.npy` files
- Location: `/pscratch/sd/n/ndwang/frequency_maps/`
- Preprocessing: Min-max normalized, optionally log-transformed

## Experiment Tracking

`EXPERIMENTS.md` is the running log of all training runs and their conclusions. **Always update it:**
- **Before** launching a new scan: write a prose section explaining the motivation, what question is being asked, what you expect to see and why, and what the result would imply either way
- **After** runs complete: record the results and write a detailed analysis — what the numbers mean, whether the outcome matched expectations, what it implies about the model or training dynamics, and what to try next

## Conventions

- Config overrides use dot notation: `model.latent_dim=64`
- Run outputs saved to `runs/<run_name>/` with config.yaml for reproducibility
- Run names: `latent{dim}_beta{beta}_{YYMMDD}_{HHMM}` (auto-generated with timestamp)
- SLURM logs go to `logs/` directory (must exist before job submission)
- SLURM scripts use GNU parallel for multi-GPU sweeps

## Environment

```bash
ml load conda
conda activate vae
```
