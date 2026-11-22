# ruff: noqa: E402
"""WaferVision AI: Semiconductor Wafer Defect Detection & Classification Web Application."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.dashboard import render_performance_dashboard
from app.components.prediction import render_prediction_summary, render_probability_chart
from app.components.upload import render_input_selector
from app.components.visualization import render_explainability_panel
from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.image_utils import wafer_to_pil
from app.utils.model_loader import (
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    load_frozen_model,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WaferVision AI | Semiconductor Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Industrial Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Global font & background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #080c14;
        border-right: 1px solid #1e293b;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background-color: #121826;
        border: 1px solid #1e293b;
        border-radius: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading and verifying CustomWaferCNN model...")
def get_model():
    """Cached loader for the frozen CustomWaferCNN baseline model."""
    model, metadata, device = load_frozen_model(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        verify_hash=True,
    )
    return model, metadata, device


def main():
    # Load model singleton
    try:
        model, metadata, device = get_model()
    except Exception as e:
        st.error(f"❌ Critical Error loading model checkpoint: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🔬 WaferVision AI")
        st.caption("Semiconductor Defect Classification & Explainability")
        st.markdown("---")

        page = st.radio(
            "Navigation:",
            options=[
                "🛰️ Wafer Analysis & XAI",
                "📊 Benchmark Performance",
                "ℹ️ Architecture & Fab Guidelines",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        st.markdown("**Model:** `CustomWaferCNN`")
        st.markdown("**Parameters:** `520,681` (Frozen)")
        st.markdown(f"**Device:** `{device}`")
        st.markdown("**Checkpoint:** `Epoch 24 (Macro F1: 0.8861)`")

        with st.expander("🔐 Checkpoint Hash Verification"):
            st.code(EXPECTED_CHECKPOINT_SHA256, language="text")
            st.caption("Verified match against sealed benchmark artifact.")

        st.markdown("---")
        st.caption(
            "Industrial AI deployment for semiconductor yield engineering. "
            "Trained on WM-811K benchmark dataset."
        )

    # -------------------------------------------------------------------------
    # Page 1: Wafer Analysis & Explainability
    # -------------------------------------------------------------------------
    if page == "🛰️ Wafer Analysis & XAI":
        st.title("🛰️ Semiconductor Wafer Defect Analysis")
        st.markdown(
            "Upload a wafer map or select a demonstration sample to classify spatial defect patterns "
            "and inspect convolutional layer activations via **Grad-CAM**."
        )

        # Input selection (Upload or Sample)
        wafer_map, source_desc, stats = render_input_selector()

        if wafer_map is not None and stats is not None:
            st.markdown("---")

            # Preprocess via Method C letterbox padding
            preprocessed_map, tensor_batch = WaferPreprocessingService.preprocess_for_inference(
                wafer_map=wafer_map,
                target_size=(64, 64),
            )

            # Execute model inference
            result = WaferInferenceService.predict(
                model=model,
                input_tensor=tensor_batch,
                device=device,
            )

            # Two-Column Layout: Visual Input + Classification Diagnostics
            col_left, col_right = st.columns([1, 1.4])

            with col_left:
                st.markdown("### 🖨️ Input Wafer Representation")
                pil_display = wafer_to_pil(preprocessed_map, scale=4)
                st.image(
                    pil_display,
                    caption=f"{source_desc} (Method C Letterbox: 64×64 dies)",
                    use_container_width=True,
                )

                # Die statistics
                st.markdown("**Silicon Die Statistics:**")
                stat_col1, stat_col2 = st.columns(2)
                stat_col1.metric("Active Silicon Dies", f"{stats['total_active_dies']:,}")
                stat_col1.metric("Functional Dies", f"{stats['normal_dies']:,}")
                stat_col2.metric("Defective Dies", f"{stats['defective_dies']:,}")
                stat_col2.metric("Defect Rate", f"{stats['defect_rate_pct']:.2f}%")

            with col_right:
                # Prediction cards
                render_prediction_summary(result)
                # Probability distribution chart
                render_probability_chart(
                    probabilities=result["probabilities"],
                    predicted_class=result["predicted_class"],
                )

            st.markdown("---")

            # Grad-CAM Explainability Section
            render_explainability_panel(
                model=model,
                input_tensor=tensor_batch,
                preprocessed_map=preprocessed_map,
                predicted_class=result["predicted_class"],
                device=device,
            )

    # -------------------------------------------------------------------------
    # Page 2: Benchmark Performance & Metrics
    # -------------------------------------------------------------------------
    elif page == "📊 Benchmark Performance":
        render_performance_dashboard()

    # -------------------------------------------------------------------------
    # Page 3: Architecture & Fab Guidelines
    # -------------------------------------------------------------------------
    elif page == "ℹ️ Architecture & Fab Guidelines":
        st.title("ℹ️ Architecture, Training Methodology & Fab Guidelines")

        st.markdown(
            """
            ### 🏛️ CustomWaferCNN Architecture Overview
            The classification engine uses **CustomWaferCNN**, an ultra-lightweight 4-block convolutional neural network
            tailored specifically for discrete wafer die grids:

            - **Total Parameters:** **520,681** ($21.5\times$ smaller than standard ResNet-18).
            - **Input Layer:** Single-channel Float32 (`[B, 1, 64, 64]`) with discrete die states $\\{0: \text{exterior}, 1: \text{normal}, 2: \text{defective}\\}$.
            - **Convolutional Backbone:** 4 residual conv blocks with Batch Normalization, ReLU activations, and Max Pooling ($64 \to 32 \to 16 \to 8 \to 4$).
            - **Spatial Grid Preservation:** Final feature grid preserves a **$4 \times 4$ spatial resolution** (16 cells) before Adaptive Average Pooling, enabling precise localization of thin scratch tracks and small clusters that standard $32\times$ downsampling networks blur out.
            - **Regularization:** Dual dropout stages ($p=0.25$ conv, $p=0.50$ linear head) and weight decay ($10^{-2}$).
            """
        )

        st.markdown("---")
        st.markdown("### ⚠️ Known Engineering Limitations & Human Review Protocol")
        st.warning(
            r"""
            **Operational Fab Boundaries:**
            1. **Imperfect Defect Recall on Sparse Geometries:**
               - **Scratch Recall:** On the held-out test set, the model detects **64.12%** of scratches as `Scratch` and flags an additional $12.94\%$ as other defect types ($77.06\%$ overall anomaly detection), but misses $22.94\%$ ($39 / 170$) as `none`.
               - **Loc Recall:** Localized cluster recall is **77.97%** on the test set, with an $11.89\%$ escape rate to `none`.
            2. **Grad-CAM Attribution Boundaries:** Grad-CAM produces coarse gradient heatmaps indicating which features weighted the classification; it does **not** generate pixel-accurate die segmentation masks or prove causal physical failure mechanisms.
            3. **Domain Transferability:** This model was qualified on 300mm/200mm wafers from the WM-811K benchmark. Deployment to new fabrication facilities or novel lithography nodes requires retraining qualification.
            4. **Production Decision Rule:** Automated tool interlocks or wafer scrap actions should not be triggered solely on high-confidence predictions without secondary optical review.
            """
        )

        st.markdown("---")
        st.markdown("### 📚 9 Canonical Defect Classes in WM-811K")
        st.markdown(
            """
            | Class | Physical Mechanism | Typical Fab Source |
            | :--- | :--- | :--- |
            | **`none`** | Acceptable production run | Normal operation |
            | **`Center`** | CMP slurry over-polish, center gas nozzle variations | Polishing / CVD Chamber |
            | **`Donut`** | Radial thermal annealing gradient, spin-rinse slurry pooling | RTA Chamber / Spin Dryer |
            | **`Edge-Loc`** | Edge Bead Removal (EBR) solvent splash, clamp pin stress | EBR Station / Pre-aligner |
            | **`Edge-Ring`** | RF plasma sheath edge distortion, resist bevel drift | Plasma Etch / Bevel Etch |
            | **`Loc`** | Liquid splatter, chuck vacuum pin contamination | ESC Chuck / Wet Bench |
            | **`Random`** | Airborne particulates, raw substrate dislocation defects | Cleanroom Air / Substrate |
            | **`Scratch`** | Robotic transfer arm blade contact, CMP pad abrasive grit | Wafer Robot / CMP Pad |
            | **`Near-full`** | Gross chamber contamination, gas delivery starvation | Major Tool Abort |
            """
        )


if __name__ == "__main__":
    main()
