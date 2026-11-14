"""Unit tests for wafer map input representations and encodings."""

import numpy as np
import pytest
import torch

from src.data.transforms import (
    WaferTransform,
    encode_one_hot,
    encode_single_channel,
)


@pytest.fixture
def raw_wafer_matrix() -> np.ndarray:
    """Create a sample 20x30 wafer matrix with exterior (0), normal (1), and defect (2)."""
    wm = np.zeros((20, 30), dtype=np.uint8)
    wm[5:15, 5:25] = 1  # Normal die
    wm[8:12, 10:15] = 2  # Defect die
    return wm


def test_encode_single_channel_shape_and_values(raw_wafer_matrix):
    """Verify single-channel numerical encoding produces [1, H, W] float32 tensor."""
    tensor = encode_single_channel(raw_wafer_matrix)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 20, 30)
    assert tensor.dtype == torch.float32

    # Verify discrete values match original
    unique_vals = set(tensor.unique().tolist())
    assert unique_vals == {0.0, 1.0, 2.0}


def test_encode_one_hot_shape_and_channels(raw_wafer_matrix):
    """Verify 3-channel one-hot representation produces [3, H, W] with correct channel assignments."""
    one_hot = encode_one_hot(raw_wafer_matrix)

    assert isinstance(one_hot, torch.Tensor)
    assert one_hot.shape == (3, 20, 30)
    assert one_hot.dtype == torch.float32

    # Verify mutually exclusive partition: sum across channels must be exactly 1.0 everywhere
    channel_sum = one_hot.sum(dim=0)
    assert torch.all(channel_sum == 1.0)

    # Channel 0: Exterior space (where raw == 0)
    assert torch.all(one_hot[0, :5, :] == 1.0)
    assert torch.all(one_hot[1, :5, :] == 0.0)
    assert torch.all(one_hot[2, :5, :] == 0.0)

    # Channel 1: Normal die (where raw == 1)
    assert one_hot[1, 6, 6].item() == 1.0
    assert one_hot[0, 6, 6].item() == 0.0
    assert one_hot[2, 6, 6].item() == 0.0

    # Channel 2: Defective die (where raw == 2)
    assert one_hot[2, 10, 12].item() == 1.0
    assert one_hot[0, 10, 12].item() == 0.0
    assert one_hot[1, 10, 12].item() == 0.0


def test_wafer_transform_pipeline_padding_exterior(raw_wafer_matrix):
    """Verify padding margins in transformed maps are properly encoded as exterior space."""
    # Test letterbox padding (Method C) with one-hot representation
    transform_one_hot = WaferTransform(
        preprocessing_method="method_c",
        representation="one_hot",
        target_size=(64, 64),
    )
    out_one_hot = transform_one_hot(raw_wafer_matrix)

    assert out_one_hot.shape == (3, 64, 64)
    # The top border (0:10, 0:64) is padding/exterior space
    assert torch.all(out_one_hot[0, 0:5, :] == 1.0)
    assert torch.all(out_one_hot[1, 0:5, :] == 0.0)
    assert torch.all(out_one_hot[2, 0:5, :] == 0.0)

    # Test single-channel numerical representation
    transform_single = WaferTransform(
        preprocessing_method="method_c",
        representation="single_channel",
        target_size=(64, 64),
    )
    out_single = transform_single(raw_wafer_matrix)

    assert out_single.shape == (1, 64, 64)
    assert torch.all(out_single[0, 0:5, :] == 0.0)  # Exterior padding is 0.0
