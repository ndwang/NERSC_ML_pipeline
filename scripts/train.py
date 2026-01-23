#!/usr/bin/env python
"""Main training script for VAE models."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models import VAE2D, ResidualVAE2D
from src.data import FrequencyMapDataset
from src.training import Trainer


def get_args():
    parser = argparse.ArgumentParser(description="VAE Training")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="/pscratch/sd/n/ndwang/frequency_maps/frequency_maps_minmax.npy",
        help="Path to dataset file"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="vae2d",
        choices=["vae2d", "residual"],
        help="Model architecture"
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, nargs="+", default=[32, 64, 128, 256, 512])

    # Training arguments
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.0, help="KL divergence weight")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./runs", help="Output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (auto-generated if not provided)")

    return parser.parse_args()


def main():
    args = get_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset setup
    full_dataset = FrequencyMapDataset(args.dataset)
    n_samples = len(full_dataset)
    val_size = int(args.val_split * n_samples)
    train_size = n_samples - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"Dataset: {n_samples} samples ({train_size} train, {val_size} val)")

    # Model config
    config = {
        "model": {
            "input_channels": 15,
            "hidden_channels": args.hidden_channels,
            "latent_dim": args.latent_dim,
            "input_size": 64,
            "kernel_size": 3,
            "activation": "relu",
            "batch_norm": True,
            "dropout_rate": 0.0,
            "weight_init": "kaiming_normal",
            "output_activation": "sigmoid",
            "use_reparameterization": True,
        }
    }

    # Create model
    if args.model == "vae2d":
        model = VAE2D(config)
    else:
        model = ResidualVAE2D(config)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        beta=args.beta,
        loss_type="mse",
        grad_clip=args.grad_clip,
    )

    # Generate run name if not provided
    if args.run_name is None:
        model_name = "VAE2D" if args.model == "vae2d" else "ResidualVAE2D"
        args.run_name = f"{model_name}_e{args.epochs}_B{args.beta}_lr{args.lr}_latent{args.latent_dim}"

    output_dir = Path(args.output_dir) / args.run_name

    # Train
    print(f"Starting training: {args.run_name}")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        max_steps=args.max_steps,
        save_dir=output_dir,
        model_name=args.run_name,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
