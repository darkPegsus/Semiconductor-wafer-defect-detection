"""Early stopping mechanism with metric-driven checkpoint saving."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from loguru import logger


class EarlyStopping:
    """Early stopping to prevent overfitting and save best model weights.

    Monitors a metric (e.g., 'val_macro_f1' with mode='max') and stops training
    if no improvement is observed after `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: Literal["min", "max"] = "max",
        checkpoint_path: str | Path = "models/checkpoints/best_model.pt",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def is_improvement(self, score: float) -> bool:
        """Check if score is an improvement over best_score by at least min_delta."""
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)

    def step(
        self,
        score: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate validation metric, update counter, and save checkpoint if improved.

        Args:
            score: Current epoch metric score.
            epoch: Current epoch index.
            model: PyTorch model instance.
            optimizer: Optional optimizer state to save.
            scheduler: Optional learning rate scheduler state to save.
            metadata: Additional metadata dictionary (metrics, hyperparameters).

        Returns:
            True if this epoch resulted in a new best checkpoint.
        """
        if self.is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                score=score,
                metadata=metadata,
            )
            logger.info(
                f"Validation metric improved to {score:.4f} at epoch {epoch}. "
                f"Checkpoint saved to {self.checkpoint_path}"
            )
            return True
        else:
            self.counter += 1
            logger.info(
                f"No improvement for {self.counter}/{self.patience} epochs. "
                f"(Best: {self.best_score:.4f} at epoch {self.best_epoch})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement."
                )
            return False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        epoch: int = 0,
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Serialize model state and training metadata."""
        checkpoint = {
            "epoch": epoch,
            "best_score": score,
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        """Load saved checkpoint into model and optimizer."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
