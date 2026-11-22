"""Unit tests for wafer preprocessing and input validation service."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.preprocessing_service import WaferPreprocessingService


@pytest.fixture
def synthetic_wafer_2d():
    """Create a synthetic 2D wafer array [30, 40] with values in {0, 1, 2}."""
    arr = np.zeros((30, 40), dtype=np.uint8)
    # Circle of normal dies
    y, x = np.ogrid[:30, :40]
    dist = np.sqrt((x - 20) ** 2 + (y - 15) ** 2)
    arr[dist <= 12] = 1
    # Defect cluster
    arr[dist <= 4] = 2
    return arr


def test_standardize_raw_wafer_matrix(synthetic_wafer_2d):
    """Verify matrix standardization preserves discrete values."""
    standardized = WaferPreprocessingService.standardize_raw_wafer_matrix(synthetic_wafer_2d)
    assert standardized.shape == (30, 40)
    assert standardized.dtype == np.uint8
    assert set(np.unique(standardized)).issubset({0, 1, 2})


def test_preprocess_for_inference(synthetic_wafer_2d):
    """Verify Method C preprocessing outputs [1, 1, 64, 64] tensor."""
    prep_map, tensor_batch = WaferPreprocessingService.preprocess_for_inference(
        synthetic_wafer_2d, target_size=(64, 64)
    )

    assert prep_map.shape == (64, 64)
    assert prep_map.dtype == np.uint8
    assert isinstance(tensor_batch, torch.Tensor)
    assert tensor_batch.shape == (1, 1, 64, 64)
    assert tensor_batch.dtype == torch.float32

    # Check discrete values
    unique_vals = set(np.unique(tensor_batch.numpy()))
    assert unique_vals.issubset({0.0, 1.0, 2.0})


def test_pil_to_wafer_map_rgb():
    """Verify RGB wafer map conversion correctly identifies defects and normal dies."""
    # Create RGB wafer image: dark background, gray normal dies, red defects
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[:, :] = [28, 32, 42]  # background
    rgb[10:30, 10:30] = [170, 175, 185]  # normal
    rgb[18:22, 18:22] = [235, 45, 45]  # defect

    pil_img = Image.fromarray(rgb)
    wafer_map = WaferPreprocessingService.pil_to_wafer_map(pil_img)

    assert wafer_map.shape == (40, 40)
    assert wafer_map[0, 0] == 0  # exterior
    assert wafer_map[15, 15] == 1  # normal die
    assert wafer_map[20, 20] == 2  # defect die


def test_load_wafer_from_bytes_png(synthetic_wafer_2d):
    """Verify loading from PNG bytes stream."""
    pil_img = Image.fromarray(synthetic_wafer_2d * 100)  # scale to 8-bit
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    loaded = WaferPreprocessingService.load_wafer_from_bytes(png_bytes, filename="test.png")
    assert loaded.shape == (30, 40)
    assert set(np.unique(loaded)).issubset({0, 1, 2})


def test_invalid_input_dimensions():
    """Verify rejection of zero-sized or too-small inputs."""
    too_small = Image.fromarray(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="too small"):
        WaferPreprocessingService.pil_to_wafer_map(too_small)


def test_corrupted_bytes_handling():
    """Verify graceful error on corrupted byte stream."""
    corrupted = b"NOT_A_VALID_IMAGE_DATA_BYTES"
    with pytest.raises(ValueError, match="Unsupported or corrupted image"):
        WaferPreprocessingService.load_wafer_from_bytes(corrupted, filename="bad.png")
