"""Grad-CAM explainability and 3-panel die map visualization component."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.services.explainability_service import WaferExplainabilityService
from app.utils.image_utils import figure_to_bytes
from src.data.dataset import IDX_TO_LABEL
from src.models.custom_cnn import CustomWaferCNN


def render_explainability_panel(
    model: CustomWaferCNN,
    input_tensor: Any,
    preprocessed_map: np.ndarray,
    predicted_class: str,
    device: Any,
) -> None:
    """Render 3-panel Grad-CAM visualization interface with class exploration.

    Args:
        model: Frozen CustomWaferCNN model.
        input_tensor: 4D tensor [1, 1, 64, 64].
        preprocessed_map: 2D array (64x64).
        predicted_class: Name of model-predicted class.
        device: Computing device.
    """
    st.markdown("#### 🔍 Model Explainability | Grad-CAM")

    col_desc, col_ctrl1, col_ctrl2 = st.columns([1.8, 1.2, 1.0])

    all_classes = [IDX_TO_LABEL[i] for i in range(9)]

    with col_desc:
        st.markdown(
            "<div style='font-size: 0.88rem; color: #475569; margin-top: 4px;'>"
            "Inspect which spatial regions of the wafer die grid drove convolutional activations at <code>features.conv_block4</code>."
            "</div>",
            unsafe_allow_html=True,
        )

    with col_ctrl1:
        target_explanation_class = st.selectbox(
            "Attribution Target Class:",
            options=all_classes,
            index=all_classes.index(predicted_class),
            help="Defaults to top predicted class. Select any other class to inspect counterfactual spatial evidence.",
        )

    with col_ctrl2:
        overlay_alpha = st.slider(
            "Overlay Opacity (α):",
            min_value=0.2,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Adjust blending transparency of the Grad-CAM heatmap over discrete dies.",
        )

    # Generate explanation
    with st.spinner("Generating Grad-CAM spatial activation map..."):
        explanation = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=input_tensor,
            wafer_map_2d=preprocessed_map,
            target_class=target_explanation_class,
            device=device,
            alpha=overlay_alpha,
        )

    orig_rgb = explanation["original_rgb"]
    heatmap = explanation["heatmap"]
    overlay_rgb = explanation["overlay_rgb"]
    target_name = explanation["target_class_name"]

    # 3-Panel Matplotlib Figure styled for clean light theme
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor="#ffffff")

    # Panel 1: Original Wafer
    axes[0].set_facecolor("#ffffff")
    axes[0].imshow(orig_rgb)
    axes[0].set_title(
        "1. Original Wafer Map (64×64)\nMethod C Aspect Letterbox",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=8,
    )
    axes[0].axis("off")

    # Panel 2: Heatmap
    axes[1].set_facecolor("#ffffff")
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Grad-CAM Heatmap\nTarget: {target_name} (conv_block4)",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=8,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors="#475569")

    # Panel 3: Overlay
    axes[2].set_facecolor("#ffffff")
    axes[2].imshow(overlay_rgb)
    is_target_pred = target_name == predicted_class
    title_suffix = "Top Prediction" if is_target_pred else "Counterfactual"
    axes[2].set_title(
        f"3. Die-Preserving Overlay\nTarget: {target_name} ({title_suffix})",
        fontsize=11,
        fontweight="bold",
        color="#0284c7" if is_target_pred else "#b45309",
        pad=8,
    )
    axes[2].axis("off")

    plt.tight_layout()
    st.image(figure_to_bytes(fig, dpi=180), use_container_width=True)

    # Clean Informational Note & Scientific Disclaimer
    st.markdown(
        """
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 16px; margin-top: 8px;">
            <div style="font-size: 0.85rem; color: #334155;">
                💡 <strong>What Grad-CAM Shows:</strong> Warm colors (red/yellow) indicate spatial die regions in the final convolutional block (receptive field downsampling to 8×8) that positively contributed to the model's classification score for the target class.
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                ⚠️ <strong>Limitation Notice:</strong> Grad-CAM is a post-hoc feature attribution method, not an exact die-level defect segmentation mask.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
