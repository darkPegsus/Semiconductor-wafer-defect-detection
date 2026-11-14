"""Training and evaluation engine with mixed precision, gradient scaling, and memory monitoring."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping
from src.training.metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    show_progress: bool = False,
) -> dict[str, float]:
    """Train model for a single epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function module.
        optimizer: Optimizer instance.
        device: Target execution device ('cuda' or 'cpu').
        scaler: Optional GradScaler for mixed precision.
        max_grad_norm: Maximum norm for gradient clipping.
        show_progress: Display tqdm progress bar.

    Returns:
        Dictionary containing 'loss' and 'peak_vram_mb'.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None and device.type == "cuda"

    iterator = tqdm(dataloader, desc="Training", leave=False) if show_progress else dataloader

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with Automatic Mixed Precision (AMP)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass with gradient scaling
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {"loss": float(epoch_loss), "peak_vram_mb": float(peak_vram_mb)}


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Evaluate model on validation or test dataset.

    Args:
        model: PyTorch model.
        dataloader: Evaluation DataLoader.
        criterion: Loss function module.
        device: Target execution device.
        show_progress: Display tqdm progress bar.

    Returns:
        Evaluation metrics dictionary including loss, accuracy, and macro F1.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    iterator = tqdm(dataloader, desc="Evaluating", leave=False) if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            preds = torch.argmax(logits, dim=-1)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    eval_loss = running_loss / max(1, total_samples)
    metrics = compute_metrics(y_true=all_targets, y_pred=all_preds)
    metrics["loss"] = float(round(eval_loss, 4))

    return metrics


class WaferTrainer:
    """End-to-end training manager orchestrating epochs, early stopping, and history."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        early_stopping: EarlyStopping | None = None,
        device: torch.device | str = "cuda",
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "models/checkpoints",
    ) -> None:
        if isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device_type="cuda") if self.use_amp else None
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "learning_rate": [],
            "peak_vram_mb": [],
        }

    def fit(
        self,
        epochs: int = 30,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Execute full training loop with validation and early stopping.

        Args:
            epochs: Maximum epochs to train.
            show_progress: Display progress bars.

        Returns:
            Dictionary containing complete training history.
        """
        logger.info(
            f"Starting training on device={self.device.type} for {epochs} epochs "
            f"(AMP={self.use_amp})..."
        )

        for epoch in range(1, epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Train phase
            train_results = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                max_grad_norm=self.max_grad_norm,
                show_progress=show_progress,
            )

            # Validation phase
            val_results = validate_one_epoch(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                show_progress=show_progress,
            )

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results["macro_f1"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_results["loss"])
            self.history["val_loss"].append(val_results["loss"])
            self.history["val_accuracy"].append(val_results["accuracy"])
            self.history["val_macro_f1"].append(val_results["macro_f1"])
            self.history["learning_rate"].append(current_lr)
            self.history["peak_vram_mb"].append(train_results["peak_vram_mb"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_results['loss']:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Val Acc: {val_results['accuracy']:.4f} | "
                f"Val Macro F1: {val_results['macro_f1']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"VRAM: {train_results['peak_vram_mb']:.1f} MB"
            )

            # Early stopping check
            if self.early_stopping is not None:
                score = (
                    val_results["macro_f1"]
                    if self.early_stopping.mode == "max"
                    else val_results["loss"]
                )
                self.early_stopping.step(
                    score=score,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metadata={"epoch": epoch, "val_metrics": val_results},
                )
                if self.early_stopping.should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        return self.history
