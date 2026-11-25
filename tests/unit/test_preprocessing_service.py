"""Unit tests for wafer preprocessing and input validation service."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.image_utils import get_wafer_die_statistics
from app.utils.model_loader import DEFAULT_CHECKPOINT_PATH, load_frozen_model


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


def test_decode_categorical_rgb_viridis():
    """Verify Viridis palette decoding (Purple 0, Teal 1, Yellow 2)."""
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    rgb[:, :] = [68, 1, 84]  # Purple exterior
    rgb[5:25, 5:25] = [32, 144, 140]  # Teal normal
    rgb[10:15, 10:15] = [253, 231, 36]  # Yellow defect

    decoded, palette, ratio = WaferPreprocessingService.decode_categorical_rgb(rgb)
    assert palette == "viridis"
    assert ratio == 0.0
    assert decoded[0, 0] == 0
    assert decoded[7, 7] == 1
    assert decoded[12, 12] == 2


def test_decode_categorical_rgb_high_contrast():
    """Verify High-contrast palette decoding (Black 0, Green 1, Red 2)."""
    rgb = np.zeros((30, 30, 3), dtype=np.uint8)
    rgb[:, :] = [0, 0, 0]  # Black
    rgb[5:25, 5:25] = [0, 200, 0]  # Green
    rgb[10:15, 10:15] = [200, 0, 0]  # Red

    decoded, palette, ratio = WaferPreprocessingService.decode_categorical_rgb(rgb)
    assert palette == "high_contrast_green_red"
    assert ratio == 0.0
    assert decoded[0, 0] == 0
    assert decoded[7, 7] == 1
    assert decoded[12, 12] == 2


def test_decode_categorical_rgb_rejection():
    """Verify rejection of non-wafer, arbitrary color images exceeding color tolerance."""
    # Arbitrary random colors
    np.random.seed(42)
    random_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match any recognized semiconductor"):
        WaferPreprocessingService.decode_categorical_rgb(random_img)


def test_reconstruct_die_grid_majority_voting():
    """Verify block majority voting correctly downsamples multi-pixel cells."""
    # 60x60 grid with 10x10 die blocks -> downsamples to 6x6
    pixel_map = np.zeros((60, 60), dtype=np.uint8)
    # Cell (2, 2): majority normal (1)
    pixel_map[20:30, 20:30] = 1
    # Add minor noise in the cell that should be suppressed by majority voting
    pixel_map[20:22, 20:22] = 2

    # Cell (4, 4): majority defect (2)
    pixel_map[40:50, 40:50] = 2

    reconstructed = WaferPreprocessingService.reconstruct_die_grid(pixel_map, target_grid=(6, 6))
    assert reconstructed.shape == (6, 6)
    assert reconstructed[0, 0] == 0
    assert reconstructed[2, 2] == 1  # noise 2 suppressed by majority vote
    assert reconstructed[4, 4] == 2


def test_reconstruct_die_grid_passthrough():
    """Verify that native wafer grids (<= 64x64) are preserved without modification."""
    native_arr = np.zeros((41, 33), dtype=np.uint8)
    native_arr[10, 10] = 1
    native_arr[20, 20] = 2
    res = WaferPreprocessingService.reconstruct_die_grid(native_arr)
    assert res.shape == (41, 33)
    assert np.array_equal(res, native_arr)


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


def test_near_full_640_sample_integration():
    """Critical end-to-end bug fix test: verify 640x640 Near-full sample is correctly decoded and classified."""
    sample_path = Path("tests/data/sample_640_near_full.jpg")
    if not sample_path.exists():
        pytest.skip(f"Test image {sample_path} not available.")

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    wafer_map, meta = WaferPreprocessingService.load_wafer_from_bytes(
        file_bytes=file_bytes, filename="sample_640_near_full.jpg", return_metadata=True
    )

    # 1. Verify palette decoding
    assert meta["detected_palette"] == "viridis"
    assert meta["unmatched_ratio"] < 0.05  # < 5% edge pixels unmatched

    # 2. Verify grid reconstruction shape
    assert wafer_map.shape == (64, 64)

    # 3. Verify morphology die counts (defect rate should be ~78%, NOT 0.16%)
    stats = get_wafer_die_statistics(wafer_map)
    assert stats["total_active_dies"] > 3000
    assert stats["defective_dies"] > 2000
    assert stats["defect_rate_pct"] > 70.0

    # 4. End-to-end model inference with frozen CustomWaferCNN checkpoint (if available)
    if Path(DEFAULT_CHECKPOINT_PATH).exists():
        model, _, _ = load_frozen_model(device="cpu")
        _, tensor_batch = WaferPreprocessingService.preprocess_for_inference(wafer_map)
        pred_result = WaferInferenceService.predict(
            model=model, input_tensor=tensor_batch, device="cpu"
        )

        # The prediction MUST be 'Near-full' with high confidence (> 90%)
        assert pred_result["predicted_class"] == "Near-full"
        assert pred_result["confidence"] > 0.90
