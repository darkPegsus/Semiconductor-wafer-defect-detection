"""About component providing technical architecture description and model boundaries."""

import streamlit as st

from app.utils.model_loader import DEFAULT_CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256


def render_about_page() -> None:
    """Render technical documentation, architecture specs, and known model limitations."""
    st.markdown("### ℹ️ About WaferVision AI | Technical Reference")

    st.markdown(
        """
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="margin-top: 0; color: #0f172a;">Application Summary</h4>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">
                <strong>WaferVision AI</strong> is an industrial deep learning inspection system for automated spatial pattern recognition
                on semiconductor wafer maps. It classifies discrete die grid patterns into canonical manufacturing defect classes and provides
                class activation explainability via Grad-CAM at the final convolutional feature layer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                <h4 style="margin-top: 0; color: #0f172a;">🏛️ CustomWaferCNN Architecture</h4>
                <ul style="color: #334155; font-size: 0.9rem; padding-left: 20px; line-height: 1.6;">
                    <li><strong>Total Parameters:</strong> 520,681 (Frozen baseline)</li>
                    <li><strong>Input Representation:</strong> Single-channel float tensor [B, 1, 64, 64]</li>
                    <li><strong>Preprocessing:</strong> Method C aspect-ratio-preserving letterbox padding</li>
                    <li><strong>Backbone:</strong> 4 Residual Convolutional Blocks with Batch Normalization and Mish activations</li>
                    <li><strong>Attention Mechanism:</strong> CBAM (Convolutional Block Attention Module)</li>
                    <li><strong>Spatial Feature Retention:</strong> Final feature map preserves 4×4 spatial grid (16 cells) before Adaptive Average Pooling</li>
                    <li><strong>Classification Head:</strong> Dense layer with dual dropout (p=0.25 conv, p=0.50 head)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                <h4 style="margin-top: 0; color: #0f172a;">📚 WM-811K Benchmark Dataset</h4>
                <ul style="color: #334155; font-size: 0.9rem; padding-left: 20px; line-height: 1.6;">
                    <li><strong>Total Labeled Records:</strong> 172,950 wafers across 46,393 lots</li>
                    <li><strong>Partitioning:</strong> Group-aware split with verified zero lot leakage</li>
                    <li><strong>Held-Out Test Set:</strong> 24,705 wafers from 1,537 independent lots</li>
                    <li><strong>Target Defect Classes (9 Canonical Types):</strong>
                        <code>none</code>, <code>Center</code>, <code>Donut</code>, <code>Edge-Loc</code>,
                        <code>Edge-Ring</code>, <code>Loc</code>, <code>Random</code>, <code>Scratch</code>, <code>Near-full</code>
                    </li>
                    <li><strong>Die Grid Encoding:</strong> 0 = Exterior, 1 = Normal functional die, 2 = Defective die</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Visible Model Limitations Box
    st.markdown("#### ⚠️ Known Operational Limitations & Boundaries")
    st.markdown(
        """
        <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #d97706; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #92400e; font-size: 1rem; margin-bottom: 8px;">
                Critical Model Performance Boundaries:
            </div>
            <ul style="color: #78350f; font-size: 0.9rem; padding-left: 20px; line-height: 1.6; margin-bottom: 0;">
                <li><strong>Overall Test Macro F1:</strong> The model achieves a Macro F1 score of <strong>0.8746</strong> on the held-out test partition.</li>
                <li><strong>Scratch Sensitivity Limitation:</strong> Test recall for <code>Scratch</code> is <strong>64.12%</strong>. The model undetected 39 out of 170 scratches (22.94% escape rate to 'none') due to faint or broken tracks. Visual engineering review is strongly advised for suspected scratch tracks.</li>
                <li><strong>Loc (Cluster) Sensitivity Limitation:</strong> Test recall for <code>Loc</code> clusters is <strong>77.97%</strong>, with an 11.89% clean escape rate.</li>
                <li><strong>Explainability Boundary:</strong> Grad-CAM is a coarse post-hoc attribution method based on the 8×8 receptive downsampling of <code>conv_block4</code>. It is <strong>not</strong> an exact pixel-level die defect segmentation mask.</li>
                <li><strong>Engineering Verification Required:</strong> Model predictions categorize spatial morphology; they do not establish specific equipment root causes or automatically execute lot scrap/interlock actions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Checkpoint Information Card
    st.markdown("#### 🔐 Checkpoint Integrity Information")
    st.markdown(
        f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
            <div style="font-size: 0.9rem; color: #334155;">
                <strong>Active Checkpoint:</strong> <code>{DEFAULT_CHECKPOINT_PATH}</code><br>
                <strong>Training Phase:</strong> Experiment 1 (CustomWaferCNN Standard Cross-Entropy Baseline, Epoch 24)<br>
                <strong>SHA-256 Digest:</strong> <code>{EXPECTED_CHECKPOINT_SHA256}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
