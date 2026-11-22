"""Explainability service wrapping Grad-CAM for wafer defect analysis."""

from typing import Any

import numpy as np
import torch

from app.utils.image_utils import wafer_to_rgb
from src.data.dataset import IDX_TO_LABEL, LABEL_TO_IDX
from src.explainability.grad_cam import GradCAM, overlay_heatmap_on_wafer
from src.models.custom_cnn import CustomWaferCNN

GRADCAM_DISCLAIMER = (
    "Grad-CAM highlights spatial die activations that positively influence the selected class logit. "
    "It provides a post-hoc visual attribution of class-discriminative evidence, not an exact "
    "die-level physical segmentation mask or proof of causal mechanism."
)


class WaferExplainabilityService:
    """Service to generate and format Grad-CAM visual explanations."""

    TARGET_LAYER = "features.conv_block4"

    @classmethod
    def generate_explanation(
        cls,
        model: CustomWaferCNN,
        input_tensor: torch.Tensor,
        wafer_map_2d: np.ndarray,
        target_class: int | str | None = None,
        device: torch.device | str | None = None,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Generate Grad-CAM heatmap and die-preserving overlay for a wafer map.

        Args:
            model: CustomWaferCNN model.
            input_tensor: 4D FloatTensor of shape [1, 1, 64, 64].
            wafer_map_2d: 2D uint8 wafer map of shape (64, 64).
            target_class: Integer index or class name string. If None, uses model prediction.
            device: Computing device.
            alpha: Heatmap overlay blending factor (default 0.5).

        Returns:
            Dictionary containing original RGB array, heatmap array, overlay array, and metadata.
        """
        # Resolve target class index
        if target_class is None:
            resolved_target_idx = None
            target_name = "Predicted Class"
        elif isinstance(target_class, str):
            resolved_target_idx = LABEL_TO_IDX.get(target_class, None)
            target_name = target_class
        else:
            resolved_target_idx = int(target_class)
            target_name = IDX_TO_LABEL.get(resolved_target_idx, f"Class {resolved_target_idx}")

        # Ensure tensor on target device
        target_dev = device or next(model.parameters()).device
        tensor_dev = input_tensor.to(target_dev)

        # Execute GradCAM
        with GradCAM(model=model, target_layer=cls.TARGET_LAYER, device=target_dev) as cam_engine:
            cam_tensor, pred_class_t, probs_t = cam_engine.generate_cam(
                tensor_dev, target_class=resolved_target_idx
            )

        heatmap_np = cam_tensor.squeeze().cpu().numpy()
        pred_idx = int(pred_class_t.item())

        if resolved_target_idx is None:
            target_name = IDX_TO_LABEL.get(pred_idx, f"Class {pred_idx}")

        # Generate visualizations
        original_rgb = wafer_to_rgb(wafer_map_2d)
        overlay_rgb = overlay_heatmap_on_wafer(
            wafer_map=wafer_map_2d,
            heatmap=heatmap_np,
            alpha=alpha,
            colormap="jet",
            highlight_defects=True,
        )

        return {
            "target_layer": cls.TARGET_LAYER,
            "target_class_idx": resolved_target_idx
            if resolved_target_idx is not None
            else pred_idx,
            "target_class_name": target_name,
            "heatmap": heatmap_np,
            "overlay_rgb": overlay_rgb,
            "original_rgb": original_rgb,
            "disclaimer": GRADCAM_DISCLAIMER,
        }
