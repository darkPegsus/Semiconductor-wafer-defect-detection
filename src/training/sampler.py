"""Class-balanced and weighted random mini-batch sampling for imbalanced datasets."""

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_random_sampler(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    num_classes: int = 9,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority defect classes.

    Each sample receives a sampling probability inversely proportional to its
    class frequency, balancing class representation across mini-batches.

    Args:
        labels: 1D sequence of integer class labels for all training samples.
        num_classes: Total number of defect classes (default 9).
        num_samples: Total samples to draw per epoch (defaults to len(labels)).
        replacement: Draw samples with replacement (True for oversampling).

    Returns:
        PyTorch WeightedRandomSampler instance.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    # Compute class frequencies
    class_counts = np.bincount(labels_arr, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # Guard against zero-division

    # Sample weight is inverse of class frequency
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]

    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    total_samples = num_samples if num_samples is not None else len(labels_arr)

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=total_samples,
        replacement=replacement,
    )
