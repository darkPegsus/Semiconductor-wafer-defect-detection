"""Image and wafer map rendering utilities for web visualization."""

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Canonical color palette for discrete wafer maps
WAFER_COLORMAP = {
    0: (28, 32, 42),  # Exterior background / canvas padding
    1: (170, 175, 185),  # Normal functional silicon die
    2: (235, 45, 45),  # Defective / failed silicon die
}


def wafer_to_rgb(wafer_map: np.ndarray) -> np.ndarray:
    """Convert a 2D discrete wafer array into an RGB image.

    Maps:
    - 0 -> Dark Charcoal [28, 32, 42]
    - 1 -> Normal Slate Gray [170, 175, 185]
    - 2 -> Defect Crimson [235, 45, 45]

    Args:
        wafer_map: 2D integer or float array (values in {0, 1, 2}).

    Returns:
        3D uint8 array of shape (H, W, 3).
    """
    if wafer_map.ndim != 2:
        raise ValueError(f"Expected 2D wafer map, got shape {wafer_map.shape}")

    h, w = wafer_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Discretize in case of floating point values
    int_map = np.rint(wafer_map).astype(np.int32)
    rgb[int_map == 0] = WAFER_COLORMAP[0]
    rgb[int_map == 1] = WAFER_COLORMAP[1]
    rgb[int_map >= 2] = WAFER_COLORMAP[2]

    return rgb


def wafer_to_pil(wafer_map: np.ndarray, scale: int = 4) -> Image.Image:
    """Convert 2D wafer map into a scaled PIL Image for display.

    Args:
        wafer_map: 2D discrete wafer map array.
        scale: Upscaling factor using nearest neighbor for crisp die boundaries.

    Returns:
        PIL Image object.
    """
    rgb = wafer_to_rgb(wafer_map)
    img = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    return img


def get_wafer_die_statistics(wafer_map: np.ndarray) -> dict[str, Any]:
    """Calculate summary die statistics for a wafer map.

    Args:
        wafer_map: 2D discrete wafer array.

    Returns:
        Dictionary containing counts of total dies, normal dies, defective dies, and defect rate.
    """
    int_map = np.rint(wafer_map).astype(np.int32)
    exterior_count = int(np.sum(int_map == 0))
    normal_count = int(np.sum(int_map == 1))
    defect_count = int(np.sum(int_map >= 2))
    total_active_dies = normal_count + defect_count
    defect_rate_pct = (defect_count / total_active_dies * 100.0) if total_active_dies > 0 else 0.0

    return {
        "canvas_size": list(wafer_map.shape),
        "total_active_dies": total_active_dies,
        "exterior_pixels": exterior_count,
        "normal_dies": normal_count,
        "defective_dies": defect_count,
        "defect_rate_pct": round(defect_rate_pct, 2),
    }


def figure_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Convert a Matplotlib figure into PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
