"""Grad-CAM (Gradient-weighted Class Activation Mapping) for wafer defect classification.

Implements explainability for convolutional wafer defect models:
- Target layer activation and gradient extraction via PyTorch hooks
- Global average pooling of gradients to compute channel importance weights
- Rectified linear activation combination to highlight positive class evidence
- Safe normalization handling constant or low-gradient activation maps
- Die-preserving heatmap overlay maintaining wafer morphology and defect visibility
"""

from typing import Any

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_target_layer(model: nn.Module, target_layer: nn.Module | str) -> nn.Module:
    """Resolve target layer from an nn.Module instance or hierarchical string identifier.

    Supports:
    - Direct nn.Module instance
    - Named module string path (e.g. 'block4', 'block4.block.0')
    - Standard configuration aliases (e.g. 'features.conv_block4', 'conv_block4')

    Args:
        model: PyTorch model instance.
        target_layer: Module object or string descriptor.

    Returns:
        The resolved target nn.Module.

    Raises:
        ValueError: If the target layer cannot be found in model.
    """
    if isinstance(target_layer, nn.Module):
        return target_layer

    if not isinstance(target_layer, str):
        raise TypeError(f"target_layer must be nn.Module or str, got {type(target_layer).__name__}")

    named_modules = dict(model.named_modules())

    # Exact match in named modules
    if target_layer in named_modules:
        return named_modules[target_layer]

    # Handle standard configuration alias for CustomWaferCNN
    alias_map = {
        "features.conv_block4": "block4",
        "conv_block4": "block4",
        "features.conv_block3": "block3",
        "conv_block3": "block3",
        "features.conv_block2": "block2",
        "conv_block2": "block2",
        "features.conv_block1": "block1",
        "conv_block1": "block1",
    }
    if target_layer in alias_map:
        resolved_name = alias_map[target_layer]
        if resolved_name in named_modules:
            return named_modules[resolved_name]

    # Dotted path traversal
    parts = target_layer.split(".")
    curr: Any = model
    try:
        for part in parts:
            if hasattr(curr, part):
                curr = getattr(curr, part)
            elif part.isdigit() and isinstance(curr, nn.Sequential | list):
                curr = curr[int(part)]
            else:
                raise AttributeError
        if isinstance(curr, nn.Module):
            return curr
    except (AttributeError, IndexError):
        pass

    available = [name for name, _ in model.named_modules() if name]
    raise ValueError(
        f"Target layer '{target_layer}' could not be resolved in {model.__class__.__name__}. "
        f"Available named modules: {available}"
    )


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) generator.

    Attaches forward and backward hooks to a target convolutional layer to extract
    activation feature maps and class-specific gradient responses.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str = "block4",
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize GradCAM with a model and target convolutional layer.

        Args:
            model: PyTorch model (e.g. CustomWaferCNN).
            target_layer: Target module or string path (e.g. 'block4' or 'features.conv_block4').
            device: Computing device for tensors.
        """
        self.model = model
        self.target_layer_module = resolve_target_layer(model, target_layer)
        self.target_layer_name = (
            target_layer
            if isinstance(target_layer, str)
            else self.target_layer_module.__class__.__name__
        )

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self.backward_hook_handle: torch.utils.hooks.RemovableHandle | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and full backward hooks on the target layer."""
        self._remove_hooks()

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out.detach()

        def backward_hook(
            module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]
        ) -> None:
            # First element of grad_output is gradient with respect to layer output
            self.gradients = grad_output[0].detach()

        self.forward_hook_handle = self.target_layer_module.register_forward_hook(forward_hook)
        self.backward_hook_handle = self.target_layer_module.register_full_backward_hook(
            backward_hook
        )

    def _remove_hooks(self) -> None:
        """Remove active PyTorch hook handles."""
        if self.forward_hook_handle is not None:
            self.forward_hook_handle.remove()
            self.forward_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None

    def remove_hooks(self) -> None:
        """Public interface to explicitly clean up hooks and gradient references."""
        self._remove_hooks()
        self.activations = None
        self.gradients = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def __del__(self) -> None:
        self._remove_hooks()

    def generate_cam(
        self,
        x: torch.Tensor,
        target_class: int | list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate normalized 2D class activation heatmaps for input batch.

        Args:
            x: Input tensor [B, C, H, W] or [C, H, W] (will be batched automatically).
            target_class: Optional class index (or list of indices). If None, uses predicted class.

        Returns:
            Tuple of:
            - heatmaps: Normalized activation maps in [0, 1] of shape [B, H, W].
            - predicted_classes: Argmax predictions [B].
            - class_probabilities: Softmax confidence scores [B, num_classes].
        """
        # Ensure 4D tensor [B, C, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

        x = x.to(self.device)
        b, _, input_h, input_w = x.shape

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Forward pass (eval mode, enable gradients for backprop into target layer)
        with torch.set_grad_enabled(True):
            # Clone input and enable grad to preserve input graph
            x_var = x.clone().detach().requires_grad_(True)
            logits = self.model(x_var)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)

            # Determine target class per batch item
            if target_class is None:
                targets = predicted_classes.tolist()
            elif isinstance(target_class, int):
                targets = [target_class] * b
            elif isinstance(target_class, list | tuple):
                if len(target_class) != b:
                    raise ValueError(
                        f"Target class list length ({len(target_class)}) != batch size ({b})"
                    )
                targets = list(target_class)
            else:
                raise TypeError(f"Invalid target_class type: {type(target_class)}")

            cams = []

            for i in range(b):
                self.model.zero_grad(set_to_none=True)
                cls_idx = targets[i]
                score = logits[i, cls_idx]

                # Backward pass for single sample score
                # retain_graph=True if more samples remain
                retain = i < (b - 1)
                score.backward(retain_graph=retain)

                if self.activations is None or self.gradients is None:
                    raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients!")

                # Sample activations and gradients: [C, H, W]
                act_i = self.activations[i]
                grad_i = self.gradients[i]

                # Global average pooling on gradients across spatial dimensions
                weights = torch.mean(grad_i, dim=(1, 2), keepdim=True)  # [C, 1, 1]

                # Linear combination of forward activation maps
                cam_i = torch.sum(weights * act_i, dim=0, keepdim=True)  # [1, H, W]

                # Apply ReLU: only positive influence on the target class
                cam_i = F.relu(cam_i)

                # Upsample to original input spatial resolution
                cam_i = F.interpolate(
                    cam_i.unsqueeze(0),
                    size=(input_h, input_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # [1, input_h, input_w]

                # Safe normalization to [0, 1]
                min_val = cam_i.min()
                max_val = cam_i.max()
                delta = max_val - min_val

                if delta > 1e-8 and torch.isfinite(delta):
                    cam_i = (cam_i - min_val) / delta
                else:
                    # Constant or zero-gradient map: return safe zeros
                    cam_i = torch.zeros_like(cam_i)

                # Ensure all values are finite and strictly bounded in [0, 1]
                cam_i = torch.nan_to_num(cam_i, nan=0.0, posinf=1.0, neginf=0.0)
                cam_i = torch.clamp(cam_i, 0.0, 1.0)

                cams.append(cam_i)

            cams_tensor = torch.cat(cams, dim=0)  # [B, input_h, input_w]

        self.model.zero_grad(set_to_none=True)
        return (
            cams_tensor.detach().cpu(),
            predicted_classes.detach().cpu(),
            probabilities.detach().cpu(),
        )


def overlay_heatmap_on_wafer(
    wafer_map: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    highlight_defects: bool = True,
) -> np.ndarray:
    """Overlay Grad-CAM activation heatmap onto a discrete wafer map.

    Preserves the microscopic die morphology:
    - Normalizes discrete die matrix: 0 (Exterior), 1 (Normal), 2 (Defective)
    - Applies colorized heat intensity via chosen colormap
    - Seamlessly blends base wafer geometry with heatmap
    - Distinctly highlights defective dies (value=2) so activation never hides defects

    Args:
        wafer_map: 2D array [H, W] of raw die values {0, 1, 2}.
        heatmap: 2D array [H, W] of normalized Grad-CAM values in [0, 1].
        alpha: Heatmap blend weight in [0, 1].
        colormap: Matplotlib or OpenCV colormap name ('jet', 'turbo', 'inferno', 'viridis').
        highlight_defects: Whether to superimpose high-contrast defective dies.

    Returns:
        RGB image array [H, W, 3] of dtype uint8 in range [0, 255].
    """
    if wafer_map.ndim != 2:
        raise ValueError(f"Expected 2D wafer_map, got shape {wafer_map.shape}")
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    # Resize heatmap if shape mismatch
    if heatmap.shape != wafer_map.shape:
        heatmap = cv2.resize(
            heatmap,
            (wafer_map.shape[1], wafer_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Clean and clamp heatmap values
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    h, w = wafer_map.shape
    base_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Discrete die color coding:
    # 0 = Exterior space: dark charcoal / navy
    # 1 = Normal functional dies: neutral slate gray
    # 2 = Defective dies: high-contrast vivid crimson
    base_rgb[wafer_map == 0] = [28, 32, 42]
    base_rgb[wafer_map == 1] = [170, 175, 185]
    base_rgb[wafer_map == 2] = [235, 45, 45]

    # Convert heatmap in [0, 1] to colored RGB representation
    try:
        cmap_fn = cm.get_cmap(colormap)
    except AttributeError:
        import matplotlib.pyplot as plt

        cmap_fn = plt.get_cmap(colormap)

    heatmap_rgba = cmap_fn(heatmap)  # [H, W, 4] in float [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Dynamic alpha weighting:
    # Blend strongly where heatmap is active, keep base visible where activation is low
    alpha = np.clip(alpha, 0.0, 1.0)

    # Exterior mask: Mute heatmap activation outside the physical wafer boundary
    is_exterior = wafer_map == 0
    blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)

    # On exterior padding, retain muted dark background with subtle activation glow
    blended[is_exterior] = 0.75 * base_rgb[is_exterior].astype(np.float32) + 0.25 * heatmap_rgb[
        is_exterior
    ].astype(np.float32)

    overlay = np.clip(blended, 0, 255).astype(np.uint8)

    # High-contrast preservation of defective dies (die == 2)
    if highlight_defects:
        is_defect = wafer_map == 2
        # Overlay bright defective dies with slight border for maximum visibility
        overlay[is_defect] = [255, 30, 30]

    return overlay
