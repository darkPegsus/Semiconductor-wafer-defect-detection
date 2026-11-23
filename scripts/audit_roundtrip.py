# ruff: noqa: E402
"""Round-trip reconstruction verification on WM-811K numerical wafer maps.

Validates that rendering numerical wafer maps into supported RGB categorical palettes
and passing them through the application upload decoding and grid reconstruction
pipeline preserves exact array values and orientation.
"""

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.preprocessing_service import (
    KNOWN_CATEGORICAL_PALETTES,
    WaferPreprocessingService,
)
from src.data.dataset import load_labeled_dataset
from src.data.group_split import load_split_indices

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_SOURCES_DIR = PROJECT_ROOT / "tests" / "data" / "roundtrip_sources"
SAVED_SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def render_wafer_to_rgb(wafer_map: np.ndarray, palette_key: str) -> np.ndarray:
    """Render a 2D discrete wafer array into an RGB image using the specified palette.

    Args:
        wafer_map: 2D uint8 array with values in {0, 1, 2}.
        palette_key: Palette name key in KNOWN_CATEGORICAL_PALETTES.

    Returns:
        3D uint8 array of shape (H, W, 3).
    """
    centroids = KNOWN_CATEGORICAL_PALETTES[palette_key]["centroids"].astype(np.uint8)
    h, w = wafer_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val in (0, 1, 2):
        rgb[wafer_map == val] = centroids[val]
    return rgb


def check_orientation_preservation(original: np.ndarray, reconstructed: np.ndarray) -> bool:
    """Check whether spatial orientation (row/col order) is strictly preserved."""
    if original.shape != reconstructed.shape:
        return False
    return bool(np.array_equal(original, reconstructed))


def run_roundtrip_audit() -> dict[str, Any]:
    """Execute complete round-trip reconstruction on representative WM-811K validation wafers."""
    print("=" * 80)
    print(" WM-811K ROUND-TRIP RECONSTRUCTION AND CATEGORICAL DECODING AUDIT")
    print("=" * 80)

    # Load validation partition (never touching held-out test partition)
    dataset_path = PROJECT_ROOT / "data" / "interim" / "labeled_wafers.pkl"
    splits_path = PROJECT_ROOT / "data" / "interim" / "split_indices.json"

    df = load_labeled_dataset(dataset_path)
    splits = load_split_indices(splits_path, "group_stratified_70_15_15")
    val_df = df.iloc[splits["val"]].reset_index(drop=True)

    # Canonical classes to test
    target_classes = [
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

    palettes_to_test = list(KNOWN_CATEGORICAL_PALETTES.keys())
    audit_results: list[dict[str, Any]] = []

    print(
        f"Loaded validation set ({len(val_df):,} wafers across {val_df['lotName'].nunique():,} lots)."
    )
    print(
        f"Testing {len(target_classes)} defect classes across {len(palettes_to_test)} supported palettes."
    )
    print("-" * 80)

    for class_name in target_classes:
        matches = val_df[val_df["failureType"] == class_name]
        assert len(matches) > 0, f"No samples found for {class_name}"

        # 1. Load original numerical map
        sample_row = matches.iloc[0]
        original_map = sample_row["waferMap"].astype(np.uint8)
        orig_h, orig_w = original_map.shape
        total_cells = orig_h * orig_w
        lot_name = str(sample_row["lotName"])

        # 2. Save original array
        clean_name = class_name.lower().replace("-", "_")
        saved_npy_path = SAVED_SOURCES_DIR / f"{clean_name}_original.npy"
        np.save(saved_npy_path, original_map)

        for pal_key in palettes_to_test:
            pal_name = KNOWN_CATEGORICAL_PALETTES[pal_key]["name"]

            # 3. Render into supported RGB categorical palette (Lossless PNG)
            rgb_arr = render_wafer_to_rgb(original_map, pal_key)
            pil_img = Image.fromarray(rgb_arr, mode="RGB")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            # 4. Pass through complete upload decoding and grid reconstruction pipeline
            reconstructed_map, meta = WaferPreprocessingService.load_wafer_from_bytes(
                file_bytes=png_bytes,
                filename=f"{clean_name}_{pal_key}.png",
                return_metadata=True,
            )

            recon_h, recon_w = reconstructed_map.shape

            # 5. Compare reconstructed numerical array against original source array
            matching_cells = int(np.sum(reconstructed_map == original_map))
            mismatched_cells = int(total_cells - matching_cells)
            reconstruction_acc = (matching_cells / total_cells) * 100.0

            # Per-category mismatches
            mismatch_cat0 = int(np.sum((original_map == 0) & (reconstructed_map != 0)))
            mismatch_cat1 = int(np.sum((original_map == 1) & (reconstructed_map != 1)))
            mismatch_cat2 = int(np.sum((original_map == 2) & (reconstructed_map != 2)))

            # Orientation check
            orientation_preserved = check_orientation_preservation(original_map, reconstructed_map)
            detected_palette = meta.get("detected_palette", "")

            # Strict assertion for lossless PNG rendering
            assert orig_h == recon_h and orig_w == recon_w, (
                f"Dimension mismatch for {class_name} ({pal_key}): {orig_h}x{orig_w} vs {recon_h}x{recon_w}"
            )
            assert mismatched_cells == 0, (
                f"Mismatches found for {class_name} ({pal_key}): {mismatched_cells}/{total_cells}"
            )
            assert orientation_preserved, f"Orientation not preserved for {class_name} ({pal_key})"
            assert detected_palette == pal_key, (
                f"Palette misidentified: expected {pal_key}, got {detected_palette}"
            )

            result_entry = {
                "pattern": class_name,
                "lot": lot_name,
                "palette_key": pal_key,
                "palette_name": pal_name,
                "original_dimensions": [orig_h, orig_w],
                "reconstructed_dimensions": [recon_h, recon_w],
                "total_cells": total_cells,
                "matching_cells": matching_cells,
                "mismatched_cells": mismatched_cells,
                "reconstruction_accuracy_pct": reconstruction_acc,
                "per_category_mismatches": {
                    "cat_0_exterior": mismatch_cat0,
                    "cat_1_normal": mismatch_cat1,
                    "cat_2_defect": mismatch_cat2,
                },
                "orientation_preserved": orientation_preserved,
                "detected_palette": detected_palette,
                "unmatched_pixel_ratio": meta.get("unmatched_ratio", 0.0),
            }
            audit_results.append(result_entry)

    # Save summary report JSON
    report_json_path = SAVED_SOURCES_DIR / "roundtrip_report.json"
    with open(report_json_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nSuccessfully verified all {len(audit_results)} round-trip configurations!")
    print(f"Report saved to: {report_json_path}")
    print("=" * 80)
    return {"results": audit_results, "report_path": str(report_json_path)}


if __name__ == "__main__":
    run_roundtrip_audit()
