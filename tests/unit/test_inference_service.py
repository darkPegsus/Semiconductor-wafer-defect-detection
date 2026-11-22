"""Unit tests for wafer inference service and model loading."""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.services.explainability_service import WaferExplainabilityService
from app.services.inference_service import WaferInferenceService
from app.utils.model_loader import (
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    compute_file_sha256,
    load_frozen_model,
    verify_checkpoint_integrity,
)
from src.data.dataset import IDX_TO_LABEL


def test_checkpoint_sha256_integrity():
    """Verify the physical checkpoint matches the frozen benchmark SHA-256."""
    assert Path(DEFAULT_CHECKPOINT_PATH).exists(), f"Missing checkpoint {DEFAULT_CHECKPOINT_PATH}"
    actual_hash = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_hash}"
    )
    assert verify_checkpoint_integrity(DEFAULT_CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256) is True


def test_load_frozen_model_cpu():
    """Verify loading CustomWaferCNN on CPU freezes all weights."""
    model, metadata, device = load_frozen_model(device="cpu", verify_hash=True)
    assert device.type == "cpu"
    assert metadata["total_parameters"] == 520681
    assert metadata["trainable_parameters"] == 0
    assert not model.training

    for p in model.parameters():
        assert not p.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_frozen_model_cuda():
    """Verify loading CustomWaferCNN on CUDA."""
    model, metadata, device = load_frozen_model(device="cuda", verify_hash=True)
    assert device.type == "cuda"
    assert metadata["total_parameters"] == 520681


def test_inference_service_output():
    """Verify inference pipeline returns valid probabilities, confidence, and class name."""
    model, _, device = load_frozen_model(device="cpu", verify_hash=False)

    # Dummy wafer input [1, 1, 64, 64]
    dummy_input = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    # Put a center defect
    dummy_input[0, 0, 20:44, 20:44] = 1.0
    dummy_input[0, 0, 28:36, 28:36] = 2.0

    result = WaferInferenceService.predict(model, dummy_input, device=device)

    assert "predicted_class_idx" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "latency_ms" in result
    assert result["predicted_class"] in IDX_TO_LABEL.values()
    assert 0.0 <= result["confidence"] <= 1.0

    # Probabilities must sum to ~1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4

    # All 9 classes must be present
    assert len(result["probabilities"]) == 9


def test_inference_service_invalid_shape():
    """Verify inference rejects wrong tensor shapes."""
    model, _, device = load_frozen_model(device="cpu", verify_hash=False)
    bad_tensor = torch.zeros(1, 64, 64)  # 3D instead of 4D
    with pytest.raises(ValueError, match="Expected tensor shape"):
        WaferInferenceService.predict(model, bad_tensor, device=device)


def test_explainability_service_integration():
    """Verify Grad-CAM explainability integration produces valid heatmaps and overlays."""
    model, _, device = load_frozen_model(device="cpu", verify_hash=False)

    dummy_input = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    dummy_2d = np.zeros((64, 64), dtype=np.uint8)
    dummy_2d[20:44, 20:44] = 1
    dummy_2d[30:34, 30:34] = 2
    dummy_input[0, 0] = torch.from_numpy(dummy_2d).float()

    explanation = WaferExplainabilityService.generate_explanation(
        model=model,
        input_tensor=dummy_input,
        wafer_map_2d=dummy_2d,
        target_class="Center",
        device=device,
    )

    assert explanation["heatmap"].shape == (64, 64)
    assert explanation["overlay_rgb"].shape == (64, 64, 3)
    assert explanation["original_rgb"].shape == (64, 64, 3)
    assert explanation["target_class_name"] == "Center"
    assert 0.0 <= explanation["heatmap"].min() <= 1.0
    assert 0.0 <= explanation["heatmap"].max() <= 1.0
