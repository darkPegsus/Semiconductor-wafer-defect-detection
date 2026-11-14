"""Wafer map preprocessing and ablation implementations.

Implements three spatial transformation strategies:
1. Method A: Direct nearest-neighbor resize to 64x64 (anisotropic distortion).
2. Method B: Fixed canvas center-padding to 64x64 (preserves original die scale).
3. Method C: Aspect-ratio-preserving letterbox resize and symmetric padding (default).

All methods strictly preserve discrete ternary die values (0: exterior, 1: normal, 2: defect).
"""

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def preprocess_direct_resize(
    wafer_map: np.ndarray,
    target_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Method A: Direct nearest-neighbor resize to fixed target resolution.

    Scales width and height independently to target_size, causing geometric distortion
    on non-square wafers.

    Args:
        wafer_map: 2D numpy array of integer die values.
        target_size: (height, width) target output shape.

    Returns:
        2D uint8 numpy array of shape target_size.
    """
    th, tw = target_size
    # OpenCV resize takes (dsize = (width, height))
    resized = cv2.resize(wafer_map, (tw, th), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def preprocess_fixed_canvas_padding(
    wafer_map: np.ndarray,
    canvas_size: tuple[int, int] = (64, 64),
    pad_value: int = 0,
    oversize_strategy: Literal["crop", "resize"] = "crop",
) -> np.ndarray:
    """Method B: Embed original wafer at center of canvas with zero-padding.

    Preserves 1:1 original die grid resolution for all maps smaller than canvas.
    Handles oversized maps (>64 in either dimension) via center-crop or downscaling.

    Args:
        wafer_map: 2D numpy array of integer die values.
        canvas_size: (height, width) of the target canvas.
        pad_value: Fill value for exterior padding (default 0).
        oversize_strategy: 'crop' (center-crop) or 'resize' (scale down to fit).

    Returns:
        2D uint8 numpy array of shape canvas_size.
    """
    th, tw = canvas_size
    h, w = wafer_map.shape[:2]
    wm = wafer_map.copy()

    # Handle oversized wafer maps
    if h > th or w > tw:
        if oversize_strategy == "crop":
            # Center crop to canvas size if dimension exceeds canvas
            if h > th:
                y0 = (h - th) // 2
                wm = wm[y0 : y0 + th, :]
            if w > tw:
                x0 = (w - tw) // 2
                wm = wm[:, x0 : x0 + tw]
            h, w = wm.shape[:2]
        else:
            # Proportional downscale with nearest-neighbor
            scale = min(th / h, tw / w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            wm = cv2.resize(wm, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = wm.shape[:2]

    canvas = np.full((th, tw), fill_value=pad_value, dtype=np.uint8)
    y_off = (th - h) // 2
    x_off = (tw - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = wm
    return canvas


def preprocess_letterbox_padding(
    wafer_map: np.ndarray,
    target_size: tuple[int, int] = (64, 64),
    pad_value: int = 0,
) -> np.ndarray:
    """Method C: Aspect-ratio-preserving proportional resize with symmetric padding.

    Scales the longest edge to fit target_size using nearest-neighbor interpolation,
    then pads the shorter dimension symmetrically with pad_value.

    Args:
        wafer_map: 2D numpy array of integer die values.
        target_size: (height, width) of target canvas.
        pad_value: Fill value for exterior padding (default 0).

    Returns:
        2D uint8 numpy array of shape target_size.
    """
    th, tw = target_size
    h, w = wafer_map.shape[:2]

    # Calculate uniform scale factor to preserve aspect ratio
    scale = min(th / h, tw / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Nearest-neighbor resize ensures die labels remain discrete integers
    resized = cv2.resize(wafer_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((th, tw), fill_value=pad_value, dtype=np.uint8)
    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


class WaferPreprocessor:
    """Configurable preprocessor supporting Methods A, B, and C."""

    def __init__(
        self,
        method: Literal["method_a", "method_b", "method_c"] = "method_c",
        target_size: tuple[int, int] = (64, 64),
        pad_value: int = 0,
    ) -> None:
        self.method = method
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, wafer_map: np.ndarray) -> np.ndarray:
        if self.method == "method_a":
            return preprocess_direct_resize(wafer_map, self.target_size)
        if self.method == "method_b":
            return preprocess_fixed_canvas_padding(wafer_map, self.target_size, self.pad_value)
        if self.method == "method_c":
            return preprocess_letterbox_padding(wafer_map, self.target_size, self.pad_value)
        raise ValueError(f"Unknown preprocessing method: {self.method}")


def compute_transformation_metrics(
    original_map: np.ndarray,
    processed_map: np.ndarray,
    method_name: str,
) -> dict[str, Any]:
    """Compute geometric distortion and die count preservation metrics.

    Args:
        original_map: Native 2D wafer matrix.
        processed_map: Transformed 64x64 wafer matrix.
        method_name: Identifier for the preprocessing strategy.

    Returns:
        Dictionary of quantitative preservation metrics.
    """
    h_orig, w_orig = original_map.shape[:2]
    ar_orig = float(w_orig / h_orig)

    # Compute bounding box of non-zero dies in processed map
    non_zero = np.argwhere(processed_map > 0)
    if len(non_zero) > 0:
        h_proc = int(non_zero[:, 0].max() - non_zero[:, 0].min() + 1)
        w_proc = int(non_zero[:, 1].max() - non_zero[:, 1].min() + 1)
        ar_proc = float(w_proc / h_proc)
    else:
        ar_proc = 1.0

    ar_distortion = float(abs(ar_proc - ar_orig) / ar_orig)

    # Defect die count (value == 2)
    defects_orig = int((original_map == 2).sum())
    defects_proc = int((processed_map == 2).sum())
    defect_ratio = float(defects_proc / max(1, defects_orig))

    # Normal die count (value == 1)
    normal_orig = int((original_map == 1).sum())
    normal_proc = int((processed_map == 1).sum())
    normal_ratio = float(normal_proc / max(1, normal_orig))

    # Discrete categories verification
    unique_vals = sorted(np.unique(processed_map).tolist())
    discrete_preserved = all(v in {0, 1, 2} for v in unique_vals)

    return {
        "method": method_name,
        "original_shape": [h_orig, w_orig],
        "original_aspect_ratio": round(ar_orig, 4),
        "processed_aspect_ratio": round(ar_proc, 4),
        "aspect_ratio_distortion": round(ar_distortion, 4),
        "original_defect_dies": defects_orig,
        "processed_defect_dies": defects_proc,
        "defect_expansion_ratio": round(defect_ratio, 4),
        "normal_expansion_ratio": round(normal_ratio, 4),
        "discrete_values_preserved": discrete_preserved,
        "unique_values": unique_vals,
    }


def generate_ablation_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path = "data/interim/figures",
    target_size: tuple[int, int] = (64, 64),
) -> dict[str, Any]:
    """Generate side-by-side visual comparisons of original vs. Methods A, B, C.

    Selects one representative wafer map for each of the 9 defect classes,
    computes quantitative metrics, and saves high-resolution figures.

    Args:
        df: DataFrame containing labeled wafers.
        output_dir: Output directory path.
        target_size: Target resolution tuple.

    Returns:
        Dictionary of metrics collected across the representative samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [
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

    # Semiconductor inspection colormap: 0=Navy background, 1=Steel/Green normal, 2=Bright Red defect
    cmap = mcolors.ListedColormap(["#1E222D", "#4A7C59", "#E63946"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(nrows=9, ncols=4, figsize=(14, 28))
    fig.suptitle(
        "WM-811K Preprocessing Ablation Study\n"
        "Original vs. Method A (Direct Resize), Method B (Canvas Padding), Method C (Letterbox)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    column_titles = [
        "Original Wafer Map",
        "Method A: Direct Resize (64x64)",
        "Method B: Fixed Canvas Padding (64x64)",
        "Method C: Letterbox Resize (64x64)",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=12)

    all_metrics: list[dict[str, Any]] = []

    for row_idx, cls_name in enumerate(classes):
        # Pick a representative sample with non-trivial defect count
        cls_subset = df[df["failureType"] == cls_name]
        if cls_name != "none":
            # Select sample with median defect count among samples with defects
            defect_counts = cls_subset["waferMap"].apply(lambda wm: (wm == 2).sum())
            valid_indices = defect_counts[defect_counts > 0].index
            if len(valid_indices) > 0:
                sample_idx = valid_indices[len(valid_indices) // 2]
            else:
                sample_idx = cls_subset.index[0]
        else:
            sample_idx = cls_subset.index[0]

        row = df.loc[sample_idx]
        orig_wm = row["waferMap"]
        h_orig, w_orig = orig_wm.shape[:2]

        map_a = preprocess_direct_resize(orig_wm, target_size)
        map_b = preprocess_fixed_canvas_padding(orig_wm, target_size)
        map_c = preprocess_letterbox_padding(orig_wm, target_size)

        metrics_a = compute_transformation_metrics(orig_wm, map_a, "Method A")
        metrics_b = compute_transformation_metrics(orig_wm, map_b, "Method B")
        metrics_c = compute_transformation_metrics(orig_wm, map_c, "Method C")

        all_metrics.append(
            {
                "class": cls_name,
                "lotName": row["lotName"],
                "waferIndex": int(row["waferIndex"]),
                "method_a": metrics_a,
                "method_b": metrics_b,
                "method_c": metrics_c,
            }
        )

        # Plot Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig_wm, cmap=cmap, norm=norm, interpolation="nearest")
        ax_orig.set_ylabel(
            f"{cls_name}\n({h_orig}x{w_orig})",
            fontsize=10,
            fontweight="bold",
        )
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # Plot Method A
        ax_a = axes[row_idx, 1]
        ax_a.imshow(map_a, cmap=cmap, norm=norm, interpolation="nearest")
        ax_a.set_xlabel(
            f"AR dist: {metrics_a['aspect_ratio_distortion']:.2f}\nDefect exp: {metrics_a['defect_expansion_ratio']:.1f}x",
            fontsize=8,
        )
        ax_a.set_xticks([])
        ax_a.set_yticks([])

        # Plot Method B
        ax_b = axes[row_idx, 2]
        ax_b.imshow(map_b, cmap=cmap, norm=norm, interpolation="nearest")
        ax_b.set_xlabel(
            f"AR dist: {metrics_b['aspect_ratio_distortion']:.2f}\nDefect exp: {metrics_b['defect_expansion_ratio']:.1f}x",
            fontsize=8,
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])

        # Plot Method C
        ax_c = axes[row_idx, 3]
        ax_c.imshow(map_c, cmap=cmap, norm=norm, interpolation="nearest")
        ax_c.set_xlabel(
            f"AR dist: {metrics_c['aspect_ratio_distortion']:.2f}\nDefect exp: {metrics_c['defect_expansion_ratio']:.1f}x",
            fontsize=8,
        )
        ax_c.set_xticks([])
        ax_c.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    comparison_fig_path = output_dir / "preprocessing_ablation.png"
    fig.savefig(comparison_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Preprocessing ablation comparison figure saved to {comparison_fig_path}")

    # Save quantitative report
    metrics_json_path = output_dir / "preprocessing_ablation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Preprocessing ablation metrics saved to {metrics_json_path}")

    return {"figure_path": str(comparison_fig_path), "metrics": all_metrics}


if __name__ == "__main__":
    from src.data.dataset import load_labeled_dataset

    df = load_labeled_dataset()
    results = generate_ablation_visualizations(df)
    print("Ablation visualizations generated successfully.")
