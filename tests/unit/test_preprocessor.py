"""Unit tests for wafer map preprocessing methods A, B, and C."""

import numpy as np
import pytest

from src.data.preprocessor import (
    WaferPreprocessor,
    compute_transformation_metrics,
    preprocess_direct_resize,
    preprocess_fixed_canvas_padding,
    preprocess_letterbox_padding,
)


@pytest.fixture
def sample_small_wafer() -> np.ndarray:
    """Create a sample 25x27 wafer map (aspect ratio 1.08) with defect dies."""
    wm = np.zeros((25, 27), dtype=np.uint8)
    # Circular wafer disk: radius 11
    cy, cx = 12, 13
    for y in range(25):
        for x in range(27):
            if ((y - cy) ** 2 + (x - cx) ** 2) <= 11**2:
                wm[y, x] = 1  # Normal die
    # Add defect cluster at center
    wm[11:14, 12:15] = 2  # Defect dies (9 dies)
    return wm


@pytest.fixture
def sample_oversized_wafer() -> np.ndarray:
    """Create a sample 80x70 wafer map exceeding 64x64 canvas."""
    wm = np.zeros((80, 70), dtype=np.uint8)
    wm[10:70, 10:60] = 1
    wm[35:45, 30:40] = 2
    return wm


def test_method_a_direct_resize(sample_small_wafer):
    """Test Method A direct resize output shape, dtype, and die values."""
    out = preprocess_direct_resize(sample_small_wafer, (64, 64))

    assert out.shape == (64, 64)
    assert out.dtype == np.uint8
    unique_vals = set(np.unique(out))
    assert unique_vals.issubset({0, 1, 2})
    assert 2 in unique_vals, "Defect dies must be preserved"


def test_method_b_fixed_canvas_padding(sample_small_wafer):
    """Test Method B zero-padding preserves exact 1:1 die counts for small wafers."""
    out = preprocess_fixed_canvas_padding(sample_small_wafer, (64, 64), pad_value=0)

    assert out.shape == (64, 64)
    assert out.dtype == np.uint8
    unique_vals = set(np.unique(out))
    assert unique_vals.issubset({0, 1, 2})

    # Exact defect count must be preserved (1:1 scale without resize)
    orig_defects = (sample_small_wafer == 2).sum()
    proc_defects = (out == 2).sum()
    assert orig_defects == proc_defects

    # Exact normal die count must also be preserved
    orig_normal = (sample_small_wafer == 1).sum()
    proc_normal = (out == 1).sum()
    assert orig_normal == proc_normal


def test_method_b_oversized_handling(sample_oversized_wafer):
    """Test Method B handles oversized wafer maps gracefully with crop and resize."""
    # Strategy 1: Crop
    out_crop = preprocess_fixed_canvas_padding(
        sample_oversized_wafer, (64, 64), oversize_strategy="crop"
    )
    assert out_crop.shape == (64, 64)
    assert out_crop.dtype == np.uint8
    assert set(np.unique(out_crop)).issubset({0, 1, 2})

    # Strategy 2: Resize
    out_resize = preprocess_fixed_canvas_padding(
        sample_oversized_wafer, (64, 64), oversize_strategy="resize"
    )
    assert out_resize.shape == (64, 64)
    assert out_resize.dtype == np.uint8
    assert set(np.unique(out_resize)).issubset({0, 1, 2})


def test_method_c_letterbox_padding(sample_small_wafer):
    """Test Method C preserves aspect ratio and outputs 64x64."""
    out = preprocess_letterbox_padding(sample_small_wafer, (64, 64), pad_value=0)

    assert out.shape == (64, 64)
    assert out.dtype == np.uint8
    unique_vals = set(np.unique(out))
    assert unique_vals.issubset({0, 1, 2})

    # Metrics
    metrics = compute_transformation_metrics(sample_small_wafer, out, "Method C")
    assert metrics["discrete_values_preserved"] is True
    # Distortion must be close to 0 (aspect-ratio preserving)
    assert metrics["aspect_ratio_distortion"] < 0.10


def test_wafer_preprocessor_class(sample_small_wafer):
    """Verify WaferPreprocessor callable interface for all 3 methods."""
    for method in ["method_a", "method_b", "method_c"]:
        preprocessor = WaferPreprocessor(method=method, target_size=(64, 64))
        result = preprocessor(sample_small_wafer)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1, 2})

    with pytest.raises(ValueError, match="Unknown preprocessing method"):
        bad_preprocessor = WaferPreprocessor(method="unknown_method")  # type: ignore
        bad_preprocessor(sample_small_wafer)
