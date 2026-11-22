"""Input handling component supporting file uploads and sample wafer selection."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.image_utils import get_wafer_die_statistics


def render_input_selector() -> tuple[np.ndarray | None, str, dict[str, Any] | None]:
    """Render the dual-mode wafer input interface (Upload vs. Demo Samples).

    Returns:
        Tuple of (wafer_map_array, source_description, metadata_dict).
    """
    st.markdown("### 📥 Select or Upload Wafer Map")

    input_mode = st.radio(
        "Choose Input Method:",
        options=["✨ Demonstration Sample Wafers", "📤 Upload Custom Wafer Map"],
        index=0,
        horizontal=True,
        help="Select a curated benchmark sample or upload your own 2D wafer map image.",
    )

    sample_dir = Path("app/assets/sample_wafers")
    manifest_path = sample_dir / "manifest.json"

    # -------------------------------------------------------------------------
    # Mode B: Demonstration Sample Wafers
    # -------------------------------------------------------------------------
    if input_mode == "✨ Demonstration Sample Wafers":
        if not manifest_path.exists():
            st.error(
                f"Sample manifest not found at {manifest_path}. Please generate samples first."
            )
            return None, "", None

        with open(manifest_path) as f:
            manifest = json.load(f)

        class_options = list(manifest.keys())
        selected_class = st.selectbox(
            "Select Canonical Wafer Defect Pattern:",
            options=class_options,
            index=7,  # Default to Scratch (Class 7)
            help="Choose from 9 representative validation wafer samples from the WM-811K benchmark.",
        )

        sample_info = manifest[selected_class]
        img_path = sample_dir / sample_info["filename"]

        if not img_path.exists():
            st.error(f"Sample image {img_path} not found.")
            return None, "", None

        pil_img = Image.open(img_path)
        wafer_map = WaferPreprocessingService.pil_to_wafer_map(pil_img)
        stats = get_wafer_die_statistics(wafer_map)

        st.caption(
            f"ℹ️ **Selected Sample:** `{selected_class}` | Original Grid: `{wafer_map.shape[0]}×{wafer_map.shape[1]}` | "
            f"Lot: `{sample_info.get('lot_name', 'Validation')}` (Demonstration Sample)"
        )

        return wafer_map, f"Sample: {selected_class} ({sample_info.get('lot_name', '')})", stats

    # -------------------------------------------------------------------------
    # Mode A: Upload Custom Wafer Map
    # -------------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Wafer Map File (PNG, JPG, BMP, or NPY):",
        type=["png", "jpg", "jpeg", "bmp", "npy"],
        help="Upload a single-channel or RGB wafer map. Formats supported: PNG, JPG, BMP, NPY.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image file or choose 'Demonstration Sample Wafers' above.")
        return None, "", None

    try:
        wafer_map = WaferPreprocessingService.load_wafer_from_bytes(
            file_bytes=uploaded_file.getvalue(),
            filename=uploaded_file.name,
        )
        stats = get_wafer_die_statistics(wafer_map)

        st.success(
            f"✅ Successfully loaded `{uploaded_file.name}` "
            f"({wafer_map.shape[0]}×{wafer_map.shape[1]} dies, {stats['total_active_dies']:,} active silicon dies)."
        )
        return wafer_map, f"Upload: {uploaded_file.name}", stats

    except Exception as e:
        st.error(f"❌ Input validation error: {e}")
        st.warning(
            "Expected format: A 2D wafer array or image containing silicon dies "
            "(0: exterior background, 1: functional normal dies, 2: defective dies)."
        )
        return None, "", None
