"""Loss functions for semiconductor wafer defect classification under extreme class imbalance."""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: np.ndarray | torch.Tensor | list[int],
    num_classes: int = 9,
    mode: Literal["balanced", "inverse", "effective_num"] = "balanced",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute per-class weights for imbalanced classification loss functions.

    Args:
        labels: 1D array/tensor/list of class targets.
        num_classes: Total number of classes (default 9).
        mode: Weighting strategy:
            - 'balanced': n_samples / (num_classes * count) (sklearn heuristic)
            - 'inverse': 1.0 / count (normalized)
            - 'effective_num': (1 - beta) / (1 - beta^count) (Cui et al., CVPR 2019)
        beta: Parameter for effective number of samples weighting.

    Returns:
        1D torch.FloatTensor of shape (num_classes,) containing class weights.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = np.sum(labels_arr == c)

    # Avoid zero division
    counts = np.maximum(counts, 1.0)
    total_samples = len(labels_arr)

    if mode == "balanced":
        weights = total_samples / (num_classes * counts)
    elif mode == "inverse":
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes
    elif mode == "effective_num":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")

    return torch.tensor(weights, dtype=torch.float32)


class MultiClassFocalLoss(nn.Module):
    """Multi-Class Focal Loss for addressing extreme wafer class imbalance.

    Formulation:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground-truth class.

    Attributes:
        gamma: Focusing parameter (default 2.0). Down-weights easy majority examples.
        alpha: Optional per-class weight tensor [num_classes] or None.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Multi-Class Focal Loss from raw logits and integer target labels.

        Args:
            logits: Unnormalized model outputs [B, C].
            targets: Integer ground-truth class indices [B].

        Returns:
            Computed scalar focal loss.
        """
        # Numerically stable log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]

        # Gather target probabilities
        targets_expanded = targets.view(-1, 1)  # [B, 1]
        log_p_t = log_probs.gather(dim=1, index=targets_expanded).squeeze(-1)  # [B]
        p_t = probs.gather(dim=1, index=targets_expanded).squeeze(-1)  # [B]

        # Focal modulating factor: (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)  # [B]
        focal_loss = -modulating_factor * log_p_t  # [B]

        # Apply class weights alpha if configured
        if self.alpha is not None:
            alpha_weights = self.alpha.to(logits.device)
            alpha_t = alpha_weights.gather(dim=0, index=targets)  # [B]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def get_loss_function(
    loss_name: Literal["cross_entropy", "weighted_cross_entropy", "focal_loss"],
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory creating loss functions for independent imbalance experiments.

    Args:
        loss_name: 'cross_entropy', 'weighted_cross_entropy', or 'focal_loss'.
        class_weights: Optional 1D tensor of class weights [num_classes].
        gamma: Focusing parameter for focal loss.

    Returns:
        Configured PyTorch loss Module.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_cross_entropy":
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_cross_entropy")
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal_loss":
        return MultiClassFocalLoss(gamma=gamma, alpha=class_weights)
    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")
