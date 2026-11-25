# ruff: noqa: E402
"""WaferVision AI: Semiconductor Wafer Defect Detection & Classification Inspection Tool."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.about import render_about_page
from app.components.dashboard import render_performance_dashboard
from app.components.header import render_inspection_header
from app.components.prediction import (
    render_inspection_notes,
    render_prediction_summary,
    render_probability_chart,
)
from app.components.upload import render_input_selector
from app.components.visualization import render_explainability_panel
from app.config import (
    EXPECTED_CHECKPOINT_SHA256,
    get_app_config,
    run_startup_healthcheck,
)
from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import WaferPreprocessingService
from app.styles import LIGHT_INDUSTRIAL_CSS
from app.utils.image_utils import wafer_to_pil
from app.utils.model_loader import (
    load_frozen_model,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WaferVision AI | Semiconductor Defect Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Centralized Light Industrial CSS Theme
st.markdown(LIGHT_INDUSTRIAL_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Verifying checkpoint and initializing CustomWaferCNN engine...")
def get_model():
    """Cached loader for the frozen CustomWaferCNN baseline model."""
    config = get_app_config()
    health_report = run_startup_healthcheck(config)
    model, metadata, device = load_frozen_model(
        checkpoint_path=config.checkpoint_path,
        device=config.resolved_device,
        verify_hash=False,  # Already verified in health check
    )
    metadata["health_report"] = health_report
    return model, metadata, device


def main():
    # Initialize and verify model singleton
    try:
        model, metadata, device = get_model()
    except Exception as e:
        st.error(f"❌ Critical Error loading model checkpoint: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Sidebar Navigation & System Information
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 12px 0;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                    <span style="color: #0284c7;">🔬</span> WaferVision AI
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">
                    Semiconductor Defect Inspection Tool
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_page = st.radio(
            "Navigation:",
            options=[
                "🔍 Wafer Inspection",
                "📊 Model Performance",
                "ℹ️ About",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("##### ⚙️ System Status")
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; font-size: 0.82rem; color: #334155;">
                <div style="margin-bottom: 4px;"><strong>Engine:</strong> <code>CustomWaferCNN</code></div>
                <div style="margin-bottom: 4px;"><strong>Parameters:</strong> <code>520,681</code> (Frozen)</div>
                <div style="margin-bottom: 4px;"><strong>Compute Device:</strong> <code>{str(device).upper()}</code></div>
                <div style="margin-bottom: 4px;"><strong>Test Macro F1:</strong> <code>0.8746</code></div>
                <div><strong>Integrity:</strong> <span style="color: #16a34a; font-weight: 700;">● Verified</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔐 Checkpoint Hash (SHA-256)"):
            st.code(EXPECTED_CHECKPOINT_SHA256, language="text")
            st.caption("Matches frozen benchmark artifact exp1_custom_cnn_baseline_best.pt")

        st.markdown("---")
        st.caption(
            "WaferVision AI v1.0 | Semiconductor Yield Engineering\n"
            "Trained on WM-811K benchmark dataset."
        )

    # -------------------------------------------------------------------------
    # Primary Page 1: Wafer Inspection Workflow
    # -------------------------------------------------------------------------
    if nav_page == "🔍 Wafer Inspection":
        render_inspection_header(device=str(device))

        # Input selection (Demonstration sample vs Upload)
        wafer_map, source_desc, stats, sample_id = render_input_selector()

        if wafer_map is not None and stats is not None:
            st.markdown("---")

            # Input Inspection & Action Panel
            col_preview, col_action = st.columns([1.1, 1.9])

            with col_preview:
                st.markdown("##### 🖨️ Wafer Input & Grid Reconstruction")
                raw_img = stats.get("raw_image")
                pil_display = wafer_to_pil(wafer_map, scale=4)
                if raw_img is not None:
                    tab_recon, tab_orig = st.tabs(["Reconstructed Grid", "Original Input"])
                    with tab_recon:
                        st.image(
                            pil_display,
                            caption=f"Decoded Die Grid ({wafer_map.shape[0]}×{wafer_map.shape[1]} dies)",
                            use_container_width=True,
                        )
                    with tab_orig:
                        st.image(
                            raw_img,
                            caption=f"Original File ({raw_img.width}×{raw_img.height} px)",
                            use_container_width=True,
                        )
                else:
                    st.image(
                        pil_display,
                        caption=f"{source_desc} ({wafer_map.shape[0]}×{wafer_map.shape[1]} dies)",
                        use_container_width=True,
                    )

            with col_action:
                st.markdown("##### 📊 Wafer Die Morphology Summary")

                stat_col1, stat_col2, stat_col3 = st.columns(3)
                total_cells = stats.get("total_grid_cells", wafer_map.size)
                ext_cells = stats.get("exterior_cells", stats.get("exterior_pixels", 0))

                stat_col1.metric("Total Grid Cells", f"{total_cells:,}")
                stat_col1.metric("Exterior Cells", f"{ext_cells:,}")

                stat_col2.metric("Active Silicon Dies", f"{stats['total_active_dies']:,}")
                stat_col2.metric("Functional Dies", f"{stats['normal_dies']:,}")

                stat_col3.metric("Defective Dies", f"{stats['defective_dies']:,}")
                stat_col3.metric("Die Defect Rate", f"{stats['defect_rate_pct']:.2f}%")

                if "palette_name" in stats:
                    orig_dims = stats.get(
                        "original_dimensions", [wafer_map.shape[1], wafer_map.shape[0]]
                    )
                    st.caption(
                        f"🎨 **Color Palette:** {stats['palette_name']} | "
                        f"**Input:** {orig_dims[0]}×{orig_dims[1]} px → "
                        f"**Grid:** {wafer_map.shape[0]}×{wafer_map.shape[1]} dies"
                    )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # Primary Action Button
                analyze_clicked = st.button(
                    "🔍 ANALYZE WAFER",
                    type="primary",
                    use_container_width=True,
                    help="Execute CustomWaferCNN classification and Grad-CAM spatial activation mapping.",
                )

            # Execution logic using session state
            current_active_id = st.session_state.get("active_sample_id")
            if analyze_clicked or (
                current_active_id == sample_id and "analysis_result" in st.session_state
            ):
                if analyze_clicked:
                    with st.spinner(
                        "Analyzing wafer map and computing Grad-CAM spatial activations..."
                    ):
                        # Preprocess via Method C letterbox padding
                        preprocessed_map, tensor_batch = (
                            WaferPreprocessingService.preprocess_for_inference(
                                wafer_map=wafer_map,
                                target_size=(64, 64),
                            )
                        )
                        # Execute model inference
                        result = WaferInferenceService.predict(
                            model=model,
                            input_tensor=tensor_batch,
                            device=device,
                        )
                        st.session_state["active_sample_id"] = sample_id
                        st.session_state["preprocessed_map"] = preprocessed_map
                        st.session_state["tensor_batch"] = tensor_batch
                        st.session_state["analysis_result"] = result
                else:
                    preprocessed_map = st.session_state["preprocessed_map"]
                    tensor_batch = st.session_state["tensor_batch"]
                    result = st.session_state["analysis_result"]

                st.markdown("---")

                # Classification & Distribution Section
                res_col_left, res_col_right = st.columns([1.1, 1.4])

                with res_col_left:
                    render_prediction_summary(result=result, stats=stats)
                    render_inspection_notes(result=result)

                with res_col_right:
                    render_probability_chart(
                        probabilities=result["probabilities"],
                        predicted_class=result["predicted_class"],
                    )

                st.markdown("---")

                # Dedicated Explainability Section: Grad-CAM
                render_explainability_panel(
                    model=model,
                    input_tensor=tensor_batch,
                    preprocessed_map=preprocessed_map,
                    predicted_class=result["predicted_class"],
                    device=device,
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 24px; text-align: center; margin-top: 20px;">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">
                            Wafer Map Loaded & Ready for Inspection
                        </div>
                        <div style="color: #64748b; font-size: 0.9rem; margin-top: 4px;">
                            Click the <strong>ANALYZE WAFER</strong> button above to execute inference and generate spatial Grad-CAM activation maps.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Page 2: Benchmark Performance & Historical Metrics
    # -------------------------------------------------------------------------
    elif nav_page == "📊 Model Performance":
        render_performance_dashboard()

    # -------------------------------------------------------------------------
    # Page 3: About & Technical Reference
    # -------------------------------------------------------------------------
    elif nav_page == "ℹ️ About":
        render_about_page()


if __name__ == "__main__":
    main()
