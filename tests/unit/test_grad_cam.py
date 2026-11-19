"""Unit tests for Grad-CAM explainability module."""

import numpy as np
import pytest
import torch

from src.explainability.grad_cam import (
    GradCAM,
    overlay_heatmap_on_wafer,
    resolve_target_layer,
)
from src.models.custom_cnn import CustomWaferCNN
from src.models.resnet18 import AdaptedResNet18


@pytest.fixture
def custom_cnn_model():
    """Create and return a CustomWaferCNN in eval mode."""
    model = CustomWaferCNN(in_channels=1, num_classes=9)
    model.eval()
    return model


@pytest.fixture
def resnet18_model():
    """Create and return an AdaptedResNet18 in eval mode without loading remote weights."""
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def dummy_wafer_batch():
    """Create a dummy wafer map batch [2, 1, 64, 64]."""
    torch.manual_seed(42)
    # Simulate discrete wafer with exterior(0), normal(1), defect(2)
    x = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    # Add circular normal wafer
    y, x_idx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    dist = torch.sqrt((x_idx - 32) ** 2 + (y - 32) ** 2)
    wafer_mask = dist <= 28
    x[:, 0][:, wafer_mask] = 1.0
    # Add defect cluster
    defect_mask = dist <= 8
    x[:, 0][:, defect_mask] = 2.0
    return x


def test_resolve_target_layer(custom_cnn_model):
    """Test target layer resolution with module, string name, and alias."""
    # Direct module
    resolved = resolve_target_layer(custom_cnn_model, custom_cnn_model.block4)
    assert resolved is custom_cnn_model.block4

    # Direct name
    resolved_name = resolve_target_layer(custom_cnn_model, "block4")
    assert resolved_name is custom_cnn_model.block4

    # Sub-block name
    resolved_sub = resolve_target_layer(custom_cnn_model, "block4.block.0")
    assert resolved_sub is custom_cnn_model.block4.block[0]

    # Standard alias
    resolved_alias = resolve_target_layer(custom_cnn_model, "features.conv_block4")
    assert resolved_alias is custom_cnn_model.block4

    # Invalid name raises ValueError
    with pytest.raises(ValueError, match="Target layer 'non_existent_layer' could not be resolved"):
        resolve_target_layer(custom_cnn_model, "non_existent_layer")

    # Invalid type raises TypeError
    with pytest.raises(TypeError):
        resolve_target_layer(custom_cnn_model, 12345)


def test_grad_cam_output_shape(custom_cnn_model, dummy_wafer_batch):
    """Verify Grad-CAM returns expected tensor dimensions and bounds."""
    with GradCAM(custom_cnn_model, target_layer="features.conv_block4") as cam:
        heatmaps, preds, probs = cam.generate_cam(dummy_wafer_batch)

        assert heatmaps.shape == (2, 64, 64)
        assert preds.shape == (2,)
        assert probs.shape == (2, 9)
        assert torch.all(torch.isfinite(heatmaps))
        assert heatmaps.min() >= 0.0
        assert heatmaps.max() <= 1.0


def test_grad_cam_prediction_consistency(custom_cnn_model, dummy_wafer_batch):
    """Verify GradCAM does not alter model predictions or probability scores."""
    with torch.no_grad():
        orig_logits = custom_cnn_model(dummy_wafer_batch)
        orig_probs = torch.softmax(orig_logits, dim=1)
        orig_preds = torch.argmax(orig_logits, dim=1)

    with GradCAM(custom_cnn_model, target_layer="block4") as cam:
        _, cam_preds, cam_probs = cam.generate_cam(dummy_wafer_batch)

    assert torch.equal(orig_preds, cam_preds)
    assert torch.allclose(orig_probs, cam_probs, atol=1e-5)


def test_grad_cam_hook_cleanup(custom_cnn_model):
    """Verify forward and backward hooks are removed upon exit."""
    cam = GradCAM(custom_cnn_model, target_layer="block4")
    assert cam.forward_hook_handle is not None
    assert cam.backward_hook_handle is not None

    cam.remove_hooks()
    assert cam.forward_hook_handle is None
    assert cam.backward_hook_handle is None
    assert cam.activations is None
    assert cam.gradients is None

    # Context manager cleanup test
    with GradCAM(custom_cnn_model, target_layer="block4") as cm_cam:
        assert cm_cam.forward_hook_handle is not None

    assert cm_cam.forward_hook_handle is None
    assert cm_cam.backward_hook_handle is None


def test_grad_cam_numerical_stability(custom_cnn_model):
    """Verify Grad-CAM handles constant inputs and zero activations safely."""
    # Completely zero input
    zero_input = torch.zeros(1, 1, 64, 64)

    with GradCAM(custom_cnn_model, target_layer="block4") as cam:
        heatmaps, preds, probs = cam.generate_cam(zero_input)

        assert heatmaps.shape == (1, 64, 64)
        assert torch.all(torch.isfinite(heatmaps))
        assert not torch.any(torch.isnan(heatmaps))
        assert not torch.any(torch.isinf(heatmaps))
        assert heatmaps.min() >= 0.0
        assert heatmaps.max() <= 1.0


def test_grad_cam_target_class_difference(custom_cnn_model, dummy_wafer_batch):
    """Verify explanations for different target classes produce different activation patterns."""
    with GradCAM(custom_cnn_model, target_layer="block4") as cam:
        cam_class0, _, _ = cam.generate_cam(dummy_wafer_batch[:1], target_class=0)
        cam_class1, _, _ = cam.generate_cam(dummy_wafer_batch[:1], target_class=1)

    # Class 0 and Class 1 should have different gradient attribution maps
    assert not torch.allclose(cam_class0, cam_class1, atol=1e-4)


def test_overlay_heatmap_on_wafer():
    """Verify overlay blending produces high-contrast, die-preserving visualization."""
    wafer = np.zeros((64, 64), dtype=np.uint8)
    wafer[10:54, 10:54] = 1
    wafer[30:35, 30:35] = 2  # defect

    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[28:38, 28:38] = 0.9  # high activation

    overlay = overlay_heatmap_on_wafer(wafer, heatmap, alpha=0.5, highlight_defects=True)

    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
    # Defect dies must be highlighted bright red
    assert np.array_equal(overlay[32, 32], [255, 30, 30])

    # Test error handling on bad shapes
    with pytest.raises(ValueError, match="Expected 2D wafer_map"):
        overlay_heatmap_on_wafer(np.zeros((64, 64, 3)), heatmap)

    with pytest.raises(ValueError, match="Expected 2D heatmap"):
        overlay_heatmap_on_wafer(wafer, np.zeros((64, 64, 1)))


def test_resolve_target_layer_resnet18(resnet18_model):
    """Test target layer resolution for AdaptedResNet18 using alias and dotted paths."""
    resolved_alias = resolve_target_layer(resnet18_model, "layer4")
    assert resolved_alias is resnet18_model.backbone.layer4

    resolved_dotted = resolve_target_layer(resnet18_model, "backbone.layer4")
    assert resolved_dotted is resnet18_model.backbone.layer4

    resolved_sub = resolve_target_layer(resnet18_model, "backbone.layer4.1")
    assert resolved_sub is resnet18_model.backbone.layer4[1]


def test_grad_cam_resnet18_output_shape(resnet18_model, dummy_wafer_batch):
    """Verify Grad-CAM on AdaptedResNet18 returns [B, 64, 64] heatmaps."""
    with GradCAM(resnet18_model, target_layer="layer4") as cam:
        heatmaps, preds, probs = cam.generate_cam(dummy_wafer_batch)

        assert heatmaps.shape == (2, 64, 64)
        assert preds.shape == (2,)
        assert probs.shape == (2, 9)
        assert torch.all(torch.isfinite(heatmaps))
        assert heatmaps.min() >= 0.0
        assert heatmaps.max() <= 1.0


def test_grad_cam_resnet18_prediction_consistency(resnet18_model, dummy_wafer_batch):
    """Verify Grad-CAM on AdaptedResNet18 does not alter predictions."""
    with torch.no_grad():
        orig_logits = resnet18_model(dummy_wafer_batch)
        orig_probs = torch.softmax(orig_logits, dim=1)
        orig_preds = torch.argmax(orig_logits, dim=1)

    with GradCAM(resnet18_model, target_layer="layer4") as cam:
        _, cam_preds, cam_probs = cam.generate_cam(dummy_wafer_batch)

    assert torch.equal(orig_preds, cam_preds)
    assert torch.allclose(orig_probs, cam_probs, atol=1e-5)


def test_grad_cam_resnet18_hook_cleanup(resnet18_model):
    """Verify hooks are cleanly detached on AdaptedResNet18."""
    cam = GradCAM(resnet18_model, target_layer="layer4")
    assert cam.forward_hook_handle is not None
    assert cam.backward_hook_handle is not None

    cam.remove_hooks()
    assert cam.forward_hook_handle is None
    assert cam.backward_hook_handle is None
