"""Input handling component supporting file uploads and demonstration wafer samples."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.image_utils import get_wafer_die_statistics


def render_input_selector() -> tuple[np.ndarray | None, str, dict[str, Any] | None, str]:
    """Render the dual-mode wafer input interface (Upload vs. Demo Samples).

    Returns:
        Tuple of (wafer_map_array, source_description, metadata_dict, input_identifier).
    """
    st.markdown("#### 📥 Wafer Map Input Selection")

    input_mode = st.radio(
        "Choose Inspection Input Mode:",
        options=["🔬 Demonstration Benchmark Samples", "📤 Upload Custom Wafer Map"],
        index=0,
        horizontal=True,
        help="Select a curated benchmark demonstration sample or upload your own wafer map file.",
    )

    sample_dir = Path("app/assets/sample_wafers")
    manifest_path = sample_dir / "manifest.json"

    # -------------------------------------------------------------------------
    # Mode B: Demonstration Benchmark Samples
    # -------------------------------------------------------------------------
    if input_mode == "🔬 Demonstration Benchmark Samples":
        if not manifest_path.exists():
            st.error(
                f"Sample manifest not found at {manifest_path}. Please generate samples first."
            )
            return None, "", None, ""

        with open(manifest_path) as f:
            manifest = json.load(f)

        class_options = list(manifest.keys())

        col_sel, col_info = st.columns([1.2, 1.8])

        with col_sel:
            selected_class = st.selectbox(
                "Select Canonical Pattern (Demonstration):",
                options=class_options,
                index=7,  # Default to Scratch (Class 7)
                help="Choose from 9 representative validation wafer samples from the WM-811K benchmark.",
            )

        sample_info = manifest[selected_class]
        img_path = sample_dir / sample_info["filename"]

        if not img_path.exists():
            st.error(f"Sample image {img_path} not found.")
            return None, "", None, ""

        pil_img = Image.open(img_path)
        wafer_map = WaferPreprocessingService.pil_to_wafer_map(pil_img)
        stats = get_wafer_die_statistics(wafer_map)
        sample_id = f"demo_{selected_class}_{sample_info.get('lot_name', '')}"

        with col_info:
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px; margin-top: 4px;">
                    <span class="status-badge-accent">DEMONSTRATION SAMPLE</span>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 4px;">
                        Pattern: <span style="color: #0284c7;">{selected_class}</span> | Lot: <code>{sample_info.get("lot_name", "Validation")}</code>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                        Benchmark Validation Split Wafer | Native Grid: {wafer_map.shape[0]}×{wafer_map.shape[1]} dies
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        return (
            wafer_map,
            f"Demonstration Sample: {selected_class} (Lot {sample_info.get('lot_name', 'N/A')})",
            stats,
            sample_id,
        )

    # -------------------------------------------------------------------------
    # Mode A: Upload Custom Wafer Map
    # -------------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Wafer Map File (Supported Formats: PNG, JPG, BMP, NPY):",
        type=["png", "jpg", "jpeg", "bmp", "npy"],
        help="Upload a single-channel or RGB wafer map file. Expected values: 0 (exterior), 1 (normal dies), 2 (defective dies).",
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload an image or array file above, or select 'Demonstration Benchmark Samples' to inspect preloaded wafers."
        )
        return None, "", None, ""

    try:
        wafer_map = WaferPreprocessingService.load_wafer_from_bytes(
            file_bytes=uploaded_file.getvalue(),
            filename=uploaded_file.name,
        )
        stats = get_wafer_die_statistics(wafer_map)
        sample_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"

        st.markdown(
            f"""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 10px 14px; margin-top: 6px;">
                <span class="status-badge-clean">FILE VERIFIED</span>
                <span style="font-weight: 700; color: #166534; margin-left: 8px;">{uploaded_file.name}</span>
                <span style="color: #475569; font-size: 0.85rem; margin-left: 8px;">
                    ({wafer_map.shape[0]}×{wafer_map.shape[1]} grid, {stats["total_active_dies"]:,} active dies, {stats["defect_rate_pct"]:.2f}% defect rate)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return wafer_map, f"Uploaded Wafer: {uploaded_file.name}", stats, sample_id

    except Exception as e:
        st.error(f"❌ Input validation failed: {e}")
        st.caption(
            "Expected format: 2D wafer array or image containing silicon dies (0: exterior background, 1: functional dies, 2: defective dies)."
        )
        return None, "", None, ""
