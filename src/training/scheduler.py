"""Learning rate scheduling utilities for wafer defect detection models."""

from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


def create_lr_scheduler(
    optimizer: Optimizer,
    name: Literal[
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "ReduceLROnPlateau",
        "StepLR",
    ] = "CosineAnnealingLR",
    epochs: int = 30,
    eta_min: float = 1e-6,
    warmup_epochs: int = 0,
    step_size: int = 10,
    gamma: float = 0.5,
    patience: int = 3,
    factor: float = 0.5,
) -> LRScheduler | ReduceLROnPlateau:
    """Create learning rate scheduler with optional linear warmup.

    Args:
        optimizer: PyTorch optimizer instance.
        name: Name of the scheduler.
        epochs: Total training epochs.
        eta_min: Minimum learning rate for cosine annealing.
        warmup_epochs: Number of linear warmup epochs (default 0).
        step_size: Step size for StepLR.
        gamma: Multiplicative factor for StepLR.
        patience: Epochs with no improvement before reducing LR (for Plateau).
        factor: Multiplicative factor for Plateau reduction.

    Returns:
        Configured PyTorch learning rate scheduler.
    """
    if name == "CosineAnnealingLR":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min
        )
    elif name == "CosineAnnealingWarmRestarts":
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=eta_min)
    elif name == "StepLR":
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, min_lr=eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler name: {name}")

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler
