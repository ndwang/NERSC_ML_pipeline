"""Logging callbacks for training metrics and model artifacts."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional


class LoggingCallback(ABC):
    """Abstract base class for logging callbacks.

    Keeps the Trainer framework-agnostic by defining a protocol
    for logging metrics and model artifacts.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current step/epoch number.
        """
        pass

    @abstractmethod
    def log_checkpoint_metadata(self, checkpoint_path: Path, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Log checkpoint metadata without uploading the file.

        Args:
            checkpoint_path: Path to the saved checkpoint.
            epoch: Epoch number.
            val_loss: Validation loss at this checkpoint.
            is_best: Whether this is the best model so far.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Clean up and finalize logging."""
        pass


class NoOpCallback(LoggingCallback):
    """No-op callback when logging is disabled."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_checkpoint_metadata(self, checkpoint_path: Path, epoch: int, val_loss: float, is_best: bool = False) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbCallback(LoggingCallback):
    """Weights & Biases logging callback.

    Args:
        run: W&B Run object.
    """

    def __init__(self, run):
        self.run = run

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to W&B."""
        self.run.log(metrics, step=step)

    def log_checkpoint_metadata(self, checkpoint_path: Path, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Log checkpoint metadata to W&B (file path and metrics only, no upload)."""
        metadata = {
            "checkpoint/path": str(checkpoint_path),
            "checkpoint/epoch": epoch,
            "checkpoint/val_loss": val_loss,
        }

        if is_best:
            metadata["checkpoint/best_model_path"] = str(checkpoint_path)
            # Update run summary for easy access to best model
            self.run.summary["best_checkpoint_path"] = str(checkpoint_path)
            self.run.summary["best_checkpoint_epoch"] = epoch
            self.run.summary["best_val_loss"] = val_loss

        self.run.log(metadata, step=epoch)

    def finish(self) -> None:
        """Finish the W&B run."""
        self.run.finish()
