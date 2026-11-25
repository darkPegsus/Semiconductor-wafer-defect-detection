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


@pytest.fixture(scope="module")
def cpu_frozen_model(tmp_path_factory):
    """Provide a frozen CustomWaferCNN instance on CPU.

    Uses DEFAULT_CHECKPOINT_PATH if present; otherwise creates a deterministic
    synthetic CustomWaferCNN checkpoint so unit tests remain 100% portable on CI.
    """
    if Path(DEFAULT_CHECKPOINT_PATH).exists():
        model, metadata, device = load_frozen_model(
            checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu", verify_hash=False
        )
        return model, metadata, device

    from src.models.custom_cnn import CustomWaferCNN

    tmp_dir = tmp_path_factory.mktemp("models")
    temp_ckpt = tmp_dir / "synthetic_baseline.pt"
    dummy_model = CustomWaferCNN(in_channels=1, num_classes=9)
    torch.save(
        {
            "model_state_dict": dummy_model.state_dict(),
            "epoch": 24,
            "best_score": 0.8861,
        },
        temp_ckpt,
    )

    model, metadata, device = load_frozen_model(
        checkpoint_path=temp_ckpt, device="cpu", verify_hash=False
    )
    return model, metadata, device


def test_checkpoint_sha256_integrity():
    """Verify the physical checkpoint matches the frozen benchmark SHA-256."""
    if not Path(DEFAULT_CHECKPOINT_PATH).exists():
        pytest.skip(f"Benchmark checkpoint not present in environment: {DEFAULT_CHECKPOINT_PATH}")
    actual_hash = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
    assert actual_hash == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint hash mismatch!\nExpected: {EXPECTED_CHECKPOINT_SHA256}\nActual:   {actual_hash}"
    )
    assert verify_checkpoint_integrity(DEFAULT_CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256) is True


def test_load_frozen_model_cpu(cpu_frozen_model):
    """Verify loading CustomWaferCNN on CPU freezes all weights."""
    model, metadata, device = cpu_frozen_model
    assert device.type == "cpu"
    assert metadata["total_parameters"] == 520681
    assert metadata["trainable_parameters"] == 0
    assert not model.training

    for p in model.parameters():
        assert not p.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_frozen_model_cuda(tmp_path):
    """Verify loading CustomWaferCNN on CUDA."""
    if Path(DEFAULT_CHECKPOINT_PATH).exists():
        ckpt_path = DEFAULT_CHECKPOINT_PATH
    else:
        from src.models.custom_cnn import CustomWaferCNN

        ckpt_path = tmp_path / "synthetic_cuda.pt"
        dummy_model = CustomWaferCNN(in_channels=1, num_classes=9)
        torch.save({"model_state_dict": dummy_model.state_dict()}, ckpt_path)

    model, metadata, device = load_frozen_model(
        checkpoint_path=ckpt_path, device="cuda", verify_hash=False
    )
    assert device.type == "cuda"
    assert metadata["total_parameters"] == 520681


def test_inference_service_output(cpu_frozen_model):
    """Verify inference pipeline returns valid probabilities, confidence, and class name."""
    model, _, device = cpu_frozen_model

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


def test_inference_service_invalid_shape(cpu_frozen_model):
    """Verify inference rejects wrong tensor shapes."""
    model, _, device = cpu_frozen_model
    bad_tensor = torch.zeros(1, 64, 64)  # 3D instead of 4D
    with pytest.raises(ValueError, match="Expected tensor shape"):
        WaferInferenceService.predict(model, bad_tensor, device=device)


def test_explainability_service_integration(cpu_frozen_model):
    """Verify Grad-CAM explainability integration produces valid heatmaps and overlays."""
    model, _, device = cpu_frozen_model

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
