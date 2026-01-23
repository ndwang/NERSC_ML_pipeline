"""VAE Trainer class for managing training loops."""

import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import vae_loss


class Trainer:
    """Trainer class for VAE models.

    Args:
        model: VAE model to train.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        device: Device to train on.
        beta: KL divergence weight for beta-VAE.
        loss_type: Type of reconstruction loss ('mse' or 'bce').
        grad_clip: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        beta: float = 0.0,
        loss_type: str = "mse",
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.loss_type = loss_type
        self.grad_clip = grad_clip

        self.model.to(self.device)

        self.history = {
            "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": []
        }

    def train_epoch(self, train_loader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data.
            max_steps: Optional maximum number of steps per epoch.

        Returns:
            Dictionary with average losses for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        loop = tqdm(train_loader, desc="Training", leave=False)
        for step, x in enumerate(loop):
            if max_steps is not None and step >= max_steps:
                break

            x = x.to(self.device)
            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, logvar, self.beta, self.loss_type
            )

            if torch.isnan(loss):
                raise ValueError(f"NaN loss detected at step {step}")

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            n_samples += batch_size

        return {
            "total": total_loss / n_samples,
            "recon": total_recon / n_samples,
            "kl": total_kl / n_samples,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dictionary with average losses.
        """
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        for x in val_loader:
            x = x.to(self.device)
            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = vae_loss(
                recon, x, mu, logvar, self.beta, self.loss_type
            )

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            n_samples += batch_size

        return {
            "total": total_loss / n_samples,
            "recon": total_recon / n_samples,
            "kl": total_kl / n_samples,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        max_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        model_name: str = "vae",
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of epochs to train.
            max_steps: Optional maximum steps per epoch.
            save_dir: Directory to save model and history.
            model_name: Base name for saved files.

        Returns:
            Training history dictionary.
        """
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, max_steps)
            val_metrics = self.validate(val_loader)

            # Update history
            self.history["train_total"].append(train_metrics["total"])
            self.history["train_recon"].append(train_metrics["recon"])
            self.history["train_kl"].append(train_metrics["kl"])
            self.history["val_total"].append(val_metrics["total"])
            self.history["val_recon"].append(val_metrics["recon"])
            self.history["val_kl"].append(val_metrics["kl"])

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_metrics['total']:.6f} | "
                f"Val: {val_metrics['total']:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        # Save model and history
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            model_path = save_dir / f"{model_name}.pth"
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")

            history_path = save_dir / f"{model_name}_history.csv"
            self._save_history(history_path, epochs)
            print(f"History saved to: {history_path}")

        return self.history

    def _save_history(self, path: Path, epochs: int) -> None:
        """Save training history to CSV file."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_total", "train_recon", "train_kl",
                "val_total", "val_recon", "val_kl"
            ])
            for ep in range(epochs):
                writer.writerow([
                    ep + 1,
                    self.history["train_total"][ep],
                    self.history["train_recon"][ep],
                    self.history["train_kl"][ep],
                    self.history["val_total"][ep],
                    self.history["val_recon"][ep],
                    self.history["val_kl"][ep],
                ])
