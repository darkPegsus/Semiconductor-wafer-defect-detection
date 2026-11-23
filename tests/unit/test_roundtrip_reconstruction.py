"""Round-trip reconstruction unit and integration tests using WM-811K wafer maps.

Verifies that original numerical wafer arrays (values {0, 1, 2}) rendered into
lossless RGB images across supported categorical palettes are reconstructed
with exact array equality, 0 mismatches, and complete orientation preservation.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.preprocessing_service import (
    KNOWN_CATEGORICAL_PALETTES,
    WaferPreprocessingService,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROUNDTRIP_SOURCES_DIR = PROJECT_ROOT / "tests" / "data" / "roundtrip_sources"

# Representative patterns explicitly requested by engineering
REPRESENTATIVE_PATTERNS = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]

SUPPORTED_PALETTES = list(KNOWN_CATEGORICAL_PALETTES.keys())


def render_to_palette_rgb(wafer_map: np.ndarray, palette_key: str) -> np.ndarray:
    """Render 2D discrete wafer array into an RGB image using the specified palette."""
    centroids = KNOWN_CATEGORICAL_PALETTES[palette_key]["centroids"].astype(np.uint8)
    h, w = wafer_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val in (0, 1, 2):
        rgb[wafer_map == val] = centroids[val]
    return rgb


@pytest.fixture(scope="module")
def source_wafer_maps() -> dict[str, np.ndarray]:
    """Load or generate representative 2D numerical wafer maps for all 9 classes."""
    maps = {}
    ROUNDTRIP_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    for pattern in REPRESENTATIVE_PATTERNS:
        clean_name = pattern.lower().replace("-", "_")
        npy_file = ROUNDTRIP_SOURCES_DIR / f"{clean_name}_original.npy"

        if npy_file.exists():
            arr = np.load(npy_file)
        else:
            # Fallback to extraction from interim labeled dataset
            from src.data.dataset import load_labeled_dataset
            from src.data.group_split import load_split_indices

            df = load_labeled_dataset(PROJECT_ROOT / "data" / "interim" / "labeled_wafers.pkl")
            splits = load_split_indices(
                PROJECT_ROOT / "data" / "interim" / "split_indices.json",
                "group_stratified_70_15_15",
            )
            val_df = df.iloc[splits["val"]].reset_index(drop=True)
            match = val_df[val_df["failureType"] == pattern].iloc[0]
            arr = match["waferMap"].astype(np.uint8)
            np.save(npy_file, arr)

        maps[pattern] = arr

    return maps


@pytest.mark.parametrize("pattern", REPRESENTATIVE_PATTERNS)
@pytest.mark.parametrize("palette_key", SUPPORTED_PALETTES)
def test_lossless_roundtrip_exact_equality(source_wafer_maps, pattern, palette_key):
    """Verify lossless 1:1 round-trip reconstruction yields exact array equality."""
    original_map = source_wafer_maps[pattern]
    orig_h, orig_w = original_map.shape
    total_cells = orig_h * orig_w

    # 1. Render to RGB using the selected palette
    rgb = render_to_palette_rgb(original_map, palette_key)
    pil_img = Image.fromarray(rgb, mode="RGB")

    # 2. Save to lossless PNG stream
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # 3. Pass through complete upload decoding and reconstruction pipeline
    clean_name = pattern.lower().replace("-", "_")
    reconstructed, meta = WaferPreprocessingService.load_wafer_from_bytes(
        file_bytes=png_bytes,
        filename=f"{clean_name}_{palette_key}.png",
        return_metadata=True,
    )

    # 4. Strict assertions
    # Dimensions
    assert reconstructed.shape == (orig_h, orig_w), (
        f"Shape mismatch: expected {(orig_h, orig_w)}, got {reconstructed.shape}"
    )

    # Exact array equality
    assert np.array_equal(reconstructed, original_map), (
        f"Array not identical for {pattern} with palette {palette_key}"
    )

    # Cell counts
    matching_cells = int(np.sum(reconstructed == original_map))
    mismatched_cells = total_cells - matching_cells
    assert matching_cells == total_cells
    assert mismatched_cells == 0

    # Per-category mismatches must be zero
    for cat in (0, 1, 2):
        cat_mismatches = int(np.sum((original_map == cat) & (reconstructed != cat)))
        assert cat_mismatches == 0, f"Category {cat} had {cat_mismatches} mismatches"

    # Orientation preservation
    assert bool(np.array_equal(reconstructed, original_map)) is True
    # Verify non-trivial asymmetry check (ensure not accidentally transposed)
    if orig_h != orig_w:
        assert not np.array_equal(reconstructed, original_map.T)

    # Palette detection
    assert meta["detected_palette"] == palette_key
    assert meta["unmatched_ratio"] == 0.0


@pytest.mark.parametrize("scale_factor", [2, 4])
def test_upscaled_roundtrip_majority_voting(source_wafer_maps, scale_factor):
    """Verify upscaled multi-pixel rendering round-trips via block majority voting."""
    # Test on Near-full and Scratch
    for pattern in ["Near-full", "Scratch"]:
        original_map = source_wafer_maps[pattern]
        orig_h, orig_w = original_map.shape

        rgb = render_to_palette_rgb(original_map, "viridis")
        pil_img = Image.fromarray(rgb, mode="RGB")

        # Upscale with nearest-neighbor (simulating multi-pixel die rendering)
        upscaled_img = pil_img.resize(
            (orig_w * scale_factor, orig_h * scale_factor),
            resample=Image.Resampling.NEAREST,
        )

        buf = io.BytesIO()
        upscaled_img.save(buf, format="PNG")

        reconstructed = WaferPreprocessingService.load_wafer_from_bytes(
            file_bytes=buf.getvalue(),
            filename=f"{pattern}_upscaled.png",
            target_grid=(orig_h, orig_w),
            return_metadata=False,
        )

        assert reconstructed.shape == (orig_h, orig_w)
        assert np.array_equal(reconstructed, original_map)
