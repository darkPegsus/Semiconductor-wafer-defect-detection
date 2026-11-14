"""Wafer map input representations, encodings, and transform pipelines.

Supports:
1. Single-channel numerical encoding: [1, H, W] with values {0.0, 1.0, 2.0}.
2. Three-channel one-hot encoding: [3, H, W] where:
   - Channel 0: Exterior space / padding (die value == 0)
   - Channel 1: Normal functional die (die value == 1)
   - Channel 2: Defective die (die value == 2)
"""

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from src.data.preprocessor import (
    preprocess_direct_resize,
    preprocess_fixed_canvas_padding,
    preprocess_letterbox_padding,
)


def encode_single_channel(
    wafer_map: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Encode 2D wafer map into single-channel float tensor [1, H, W].

    Die values are preserved as float values:
    0.0 = exterior/padding, 1.0 = normal die, 2.0 = defective die.

    Args:
        wafer_map: 2D array or tensor of shape (H, W).

    Returns:
        torch.FloatTensor of shape (1, H, W).
    """
    if isinstance(wafer_map, np.ndarray):
        tensor = torch.from_numpy(wafer_map)
    else:
        tensor = wafer_map

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    return tensor.float()


def encode_one_hot(
    wafer_map: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Encode 2D wafer map into 3-channel one-hot float tensor [3, H, W].

    Channels correspond to:
    - Channel 0: Exterior space and padding margin (die == 0)
    - Channel 1: Normal die (die == 1)
    - Channel 2: Defective die (die == 2)

    Args:
        wafer_map: 2D array or tensor of shape (H, W).

    Returns:
        torch.FloatTensor of shape (3, H, W).
    """
    if isinstance(wafer_map, np.ndarray):
        tensor = torch.from_numpy(wafer_map)
    else:
        tensor = wafer_map

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    tensor = tensor.long()
    # F.one_hot produces [H, W, 3] -> permute to [3, H, W]
    one_hot = F.one_hot(tensor, num_classes=3).permute(2, 0, 1).float()
    return one_hot


class WaferTransform:
    """Configurable pipeline combining spatial preprocessing and input encoding."""

    def __init__(
        self,
        preprocessing_method: Literal["method_a", "method_b", "method_c"] = "method_c",
        representation: Literal["single_channel", "one_hot"] = "single_channel",
        target_size: tuple[int, int] = (64, 64),
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ) -> None:
        self.preprocessing_method = preprocessing_method
        self.representation = representation
        self.target_size = target_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __call__(self, wafer_map: np.ndarray) -> torch.Tensor:
        # Step 1: Spatial preprocessing to canonical resolution
        if self.preprocessing_method == "method_a":
            processed = preprocess_direct_resize(wafer_map, self.target_size)
        elif self.preprocessing_method == "method_b":
            processed = preprocess_fixed_canvas_padding(wafer_map, self.target_size, pad_value=0)
        elif self.preprocessing_method == "method_c":
            processed = preprocess_letterbox_padding(wafer_map, self.target_size, pad_value=0)
        else:
            raise ValueError(f"Unknown preprocessing method: {self.preprocessing_method}")

        # Step 2: Augmentation on discrete 2D matrix (prior to tensor encoding)
        if self.augment:
            if np.random.rand() < self.hflip_prob:
                processed = np.fliplr(processed).copy()
            if np.random.rand() < self.vflip_prob:
                processed = np.flipud(processed).copy()

        # Step 3: Input encoding
        if self.representation == "single_channel":
            return encode_single_channel(processed)
        elif self.representation == "one_hot":
            return encode_one_hot(processed)
        else:
            raise ValueError(f"Unknown input representation: {self.representation}")
