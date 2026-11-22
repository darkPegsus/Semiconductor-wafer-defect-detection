"""Grad-CAM explainability and die map visualization component."""

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
    st.markdown("### 🔍 Explainable AI: Grad-CAM Spatial Attribution")
    st.caption(
        "Inspect which spatial regions of the wafer map drove the convolutional feature activations "
        "at the final layer (`features.conv_block4`)."
    )

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    all_classes = [IDX_TO_LABEL[i] for i in range(9)]

    with col_ctrl1:
        target_explanation_class = st.selectbox(
            "Select Attribution Target Class:",
            options=all_classes,
            index=all_classes.index(predicted_class),
            help="By default, Grad-CAM explains the model's top prediction. You can also explore counterfactual attributions for any other defect class.",
        )

    with col_ctrl2:
        overlay_alpha = st.slider(
            "Heatmap Blending Alpha:",
            min_value=0.2,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Adjust transparency of the Grad-CAM heatmap overlay.",
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

    # 3-Panel Matplotlib Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Original Wafer
    axes[0].imshow(orig_rgb)
    axes[0].set_title(
        "1. Original Wafer Map (64x64)\nMethod C Letterbox", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: Heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Grad-CAM Activation Map\nTarget: {target_name} (conv_block4)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay_rgb)
    is_target_pred = target_name == predicted_class
    title_color = "#00d4ff" if is_target_pred else "#eab308"
    axes[2].set_title(
        f"3. Die-Preserving Overlay\nTarget: {target_name} ({'Predicted' if is_target_pred else 'Counterfactual'})",
        fontsize=11,
        fontweight="bold",
        color=title_color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    st.image(figure_to_bytes(fig, dpi=180), use_container_width=True)

    # Official Scientific Disclaimer
    st.info(f"💡 **Interpretability Note:** {explanation['disclaimer']}")
