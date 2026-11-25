# ruff: noqa: E402
"""End-to-End Application Acceptance Integration Tests.

Validates the complete wafer defect detection pipeline from raw input ingestion
through categorical decoding, grid reconstruction, Method C letterbox preprocessing,
tensor creation, frozen CustomWaferCNN inference, probability distribution, and
Grad-CAM explainability spatial mapping across CPU and CUDA devices.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.explainability_service import WaferExplainabilityService
from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import (
    KNOWN_CATEGORICAL_PALETTES,
    WaferPreprocessingService,
)
from app.utils.image_utils import get_wafer_die_statistics
from app.utils.model_loader import EXPECTED_CHECKPOINT_SHA256, load_frozen_model
from src.data.dataset import IDX_TO_LABEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_DIR = PROJECT_ROOT / "tests" / "data" / "roundtrip_sources"

ALL_CLASSES = [IDX_TO_LABEL[i] for i in range(9)]
ALL_PALETTES = list(KNOWN_CATEGORICAL_PALETTES.keys())


@pytest.fixture(scope="module")
def loaded_models():
    """Load frozen CustomWaferCNN model instances on CPU and CUDA (if available)."""
    models = {}
    cpu_model, cpu_meta, _ = load_frozen_model(device="cpu")
    models["cpu"] = (cpu_model, cpu_meta)

    if torch.cuda.is_available():
        cuda_model, cuda_meta, _ = load_frozen_model(device="cuda")
        models["cuda"] = (cuda_model, cuda_meta)

    return models


# =============================================================================
# 1. Checkpoint & Frozen Model Integrity Tests
# =============================================================================


def test_checkpoint_sha256_hash_integrity():
    """Verify frozen model checkpoint SHA-256 strictly matches the benchmark artifact."""
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "exp1_custom_cnn_baseline_best.pt"
    assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"

    import hashlib

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint SHA-256 hash altered! Expected {EXPECTED_CHECKPOINT_SHA256}, got {sha256}"
    )


def test_model_parameters_frozen(loaded_models):
    """Verify that all model weights require_grad=False and parameter count is exactly 520,681."""
    model, _ = loaded_models["cpu"]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == 520681, f"Expected 520,681 params, got {total_params}"
    assert trainable_params == 0, f"Expected 0 trainable params, got {trainable_params}"


# =============================================================================
# 2. Numerical Wafer Maps (.NPY) End-to-End Ingestion
# =============================================================================


@pytest.mark.parametrize(
    "shape,label",
    [
        ((26, 26), "none"),
        ((50, 43), "Near-full"),
        ((54, 71), "Donut"),
        ((153, 187), "Edge-Ring"),
        ((212, 204), "large_synthetic"),
    ],
)
def test_numerical_wafer_maps_e2e(loaded_models, shape, label):
    """Verify ingestion of raw numerical arrays across native, non-square, and large grids."""
    model, _ = loaded_models["cpu"]
    h, w = shape

    # Create or load numerical array
    if (shape == (26, 26) or shape == (50, 43) or shape == (54, 71)) and (
        SOURCES_DIR / f"{label.lower().replace('-', '_')}_original.npy"
    ).exists():
        raw_map = np.load(SOURCES_DIR / f"{label.lower().replace('-', '_')}_original.npy")
    else:
        # Synthetic structured map
        raw_map = np.zeros((h, w), dtype=np.uint8)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt(((y - center_y) / (h / 2)) ** 2 + ((x - center_x) / (w / 2)) ** 2)
        raw_map[r <= 0.85] = 1
        raw_map[r <= 0.35] = 2

    # Encode to NPY byte stream
    buf = io.BytesIO()
    np.save(buf, raw_map)
    npy_bytes = buf.getvalue()

    # Step 1: Ingest via service
    wafer_map, meta = WaferPreprocessingService.load_wafer_from_bytes(
        file_bytes=npy_bytes, filename=f"test_{label}.npy", return_metadata=True
    )

    # Step 2: Validate die counts and dimensions
    assert wafer_map.shape == (h, w)
    stats = get_wafer_die_statistics(wafer_map)
    assert stats["total_active_dies"] > 0
    assert stats["normal_dies"] >= 0
    assert stats["defective_dies"] >= 0

    # Step 3: Preprocess for inference (Method C letterbox padding)
    prep_2d, tensor_input = WaferPreprocessingService.preprocess_for_inference(wafer_map)
    assert prep_2d.shape == (64, 64)
    assert tensor_input.shape == (1, 1, 64, 64)

    # Step 4: Model inference
    res = WaferInferenceService.predict(model=model, input_tensor=tensor_input, device="cpu")
    assert res["predicted_class"] in ALL_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 9
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-4

    # Step 5: Grad-CAM generation
    cam_res = WaferExplainabilityService.generate_explanation(
        model=model,
        input_tensor=tensor_input,
        wafer_map_2d=prep_2d,
        target_class=res["predicted_class_idx"],
        device="cpu",
    )
    assert cam_res["heatmap"].shape == (64, 64)
    assert cam_res["overlay_rgb"].shape == (64, 64, 3)


# =============================================================================
# 3. Categorical Color-Coded Images & Upscaling
# =============================================================================


@pytest.mark.parametrize("palette_key", ALL_PALETTES)
def test_palette_decoding_e2e(loaded_models, palette_key):
    """Verify complete inspection workflow across all supported categorical color palettes."""
    model, _ = loaded_models["cpu"]
    centroids = KNOWN_CATEGORICAL_PALETTES[palette_key]["centroids"].astype(np.uint8)

    # Create synthetic wafer map
    wm = np.zeros((32, 32), dtype=np.uint8)
    wm[4:28, 4:28] = 1
    wm[12:20, 12:20] = 2

    # Render to RGB
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    for v in (0, 1, 2):
        rgb[wm == v] = centroids[v]

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    wafer_map, meta = WaferPreprocessingService.load_wafer_from_bytes(
        file_bytes=buf.getvalue(), filename=f"palette_{palette_key}.png", return_metadata=True
    )

    assert meta["detected_palette"] == palette_key
    assert np.array_equal(wafer_map, wm)

    # Preprocess & inference
    prep_2d, tensor_input = WaferPreprocessingService.preprocess_for_inference(wafer_map)
    res = WaferInferenceService.predict(model=model, input_tensor=tensor_input, device="cpu")
    assert res["confidence"] > 0.0


@pytest.mark.parametrize("scale_factor", [2, 4, 10])
def test_upscaled_images_majority_voting_e2e(loaded_models, scale_factor):
    """Verify upscaled images (2x, 4x, 10x per die) reconstruct accurately via majority voting."""
    model, _ = loaded_models["cpu"]
    centroids = KNOWN_CATEGORICAL_PALETTES["viridis"]["centroids"].astype(np.uint8)

    # Base grid 26x26
    base_wm = np.zeros((26, 26), dtype=np.uint8)
    base_wm[3:23, 3:23] = 1
    base_wm[10:16, 10:16] = 2

    # Render 1:1 RGB
    rgb_base = np.zeros((26, 26, 3), dtype=np.uint8)
    for v in (0, 1, 2):
        rgb_base[base_wm == v] = centroids[v]

    pil_img = Image.fromarray(rgb_base, mode="RGB")
    upscaled = pil_img.resize(
        (26 * scale_factor, 26 * scale_factor), resample=Image.Resampling.NEAREST
    )

    buf = io.BytesIO()
    upscaled.save(buf, format="PNG")

    # Ingest without specifying target_grid (if factor is standard, or with target_grid)
    wafer_map = WaferPreprocessingService.load_wafer_from_bytes(
        file_bytes=buf.getvalue(),
        filename=f"scale_{scale_factor}.png",
        target_grid=(26, 26),
    )

    assert wafer_map.shape == (26, 26)
    assert np.array_equal(wafer_map, base_wm)

    # Preprocess and infer
    _, tensor_input = WaferPreprocessingService.preprocess_for_inference(wafer_map)
    res = WaferInferenceService.predict(model=model, input_tensor=tensor_input, device="cpu")
    assert res["predicted_class"] in ALL_CLASSES


# =============================================================================
# 4. Invalid Input Rejection & Graceful Error Handling
# =============================================================================


def test_invalid_corrupted_image_bytes():
    """Verify corrupted bytes raise clear ValueError."""
    corrupted = b"\x89PNG\r\n\x1a\nCORRUPTED_GARBAGE_PAYLOAD"
    with pytest.raises(ValueError, match="Unsupported or corrupted image file"):
        WaferPreprocessingService.load_wafer_from_bytes(corrupted, filename="corrupt.png")


def test_invalid_unsupported_file_extension():
    """Verify unsupported file types raise clear error."""
    text_data = b"ID,DIE_X,DIE_Y,STATUS\n1,10,12,PASS\n"
    with pytest.raises(ValueError, match="Unsupported or corrupted image file"):
        WaferPreprocessingService.load_wafer_from_bytes(text_data, filename="wafer.csv")


def test_invalid_empty_file():
    """Verify empty byte stream is rejected gracefully."""
    with pytest.raises(ValueError, match="Unsupported or corrupted image file"):
        WaferPreprocessingService.load_wafer_from_bytes(b"", filename="empty.png")


def test_invalid_natural_photograph_rejection():
    """Verify natural images with continuous color gradients are rejected by palette validator."""
    # Create smooth color gradient image (continuous non-wafer photo)
    y, x = np.mgrid[:100, :100]
    r = ((x / 100.0) * 255).astype(np.uint8)
    g = ((y / 100.0) * 255).astype(np.uint8)
    b = (((x + y) / 200.0) * 255).astype(np.uint8)
    photo = np.stack([r, g, b], axis=-1)

    pil_img = Image.fromarray(photo, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    with pytest.raises(ValueError, match="does not match any recognized semiconductor wafer"):
        WaferPreprocessingService.load_wafer_from_bytes(buf.getvalue(), filename="landscape.png")


def test_invalid_unsupported_palette_rejection():
    """Verify images using unsupported colors (e.g. magenta/orange/cyan random colors) are rejected."""
    np.random.seed(999)
    noise = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    pil_img = Image.fromarray(noise, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    with pytest.raises(ValueError, match="does not match any recognized semiconductor wafer"):
        WaferPreprocessingService.load_wafer_from_bytes(buf.getvalue(), filename="noise.png")


def test_invalid_sub_minimum_dimensions():
    """Verify images smaller than 8x8 pixels are rejected."""
    tiny = Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="too small"):
        WaferPreprocessingService.pil_to_wafer_map(tiny)


def test_invalid_super_maximum_dimensions():
    """Verify images larger than 2048x2048 pixels are rejected."""
    huge = Image.fromarray(np.zeros((2050, 2050, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="exceed maximum allowed dimension"):
        WaferPreprocessingService.pil_to_wafer_map(huge)


# =============================================================================
# 5. Representative Class Coverage Across All 9 Classes
# =============================================================================


@pytest.mark.parametrize("class_name", ALL_CLASSES)
def test_all_9_representative_classes_workflow(loaded_models, class_name):
    """Verify end-to-end classification and Grad-CAM on representative validation samples for all 9 classes."""
    model, _ = loaded_models["cpu"]
    clean_name = class_name.lower().replace("-", "_")
    npy_path = SOURCES_DIR / f"{clean_name}_original.npy"

    if not npy_path.exists():
        pytest.skip(f"Fixture {npy_path} not found.")

    raw_map = np.load(npy_path)

    # Complete pipeline
    stats = get_wafer_die_statistics(raw_map)
    assert stats["total_active_dies"] > 0

    prep_2d, tensor_input = WaferPreprocessingService.preprocess_for_inference(raw_map)
    assert tensor_input.shape == (1, 1, 64, 64)

    res = WaferInferenceService.predict(model=model, input_tensor=tensor_input, device="cpu")
    assert res["predicted_class"] in ALL_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0

    cam_res = WaferExplainabilityService.generate_explanation(
        model=model,
        input_tensor=tensor_input,
        wafer_map_2d=prep_2d,
        target_class=res["predicted_class_idx"],
        device="cpu",
    )
    assert cam_res["heatmap"].shape == (64, 64)
    assert cam_res["overlay_rgb"].shape == (64, 64, 3)


# =============================================================================
# 6. CPU and CUDA Parity & Execution Consistency
# =============================================================================


def test_cpu_cuda_inference_and_gradcam_parity(loaded_models):
    """Verify inference probabilities and Grad-CAM heatmaps match between CPU and CUDA."""
    if "cuda" not in loaded_models:
        pytest.skip("CUDA device not available on host system.")

    cpu_model, _ = loaded_models["cpu"]
    cuda_model, _ = loaded_models["cuda"]

    sample_path = SOURCES_DIR / "near_full_original.npy"
    if not sample_path.exists():
        pytest.skip("Near-full sample fixture not available.")

    raw_map = np.load(sample_path)
    prep_2d, tensor_input = WaferPreprocessingService.preprocess_for_inference(raw_map)

    # Run on CPU
    cpu_res = WaferInferenceService.predict(
        model=cpu_model, input_tensor=tensor_input, device="cpu"
    )
    cpu_cam = WaferExplainabilityService.generate_explanation(
        model=cpu_model,
        input_tensor=tensor_input,
        wafer_map_2d=prep_2d,
        target_class=cpu_res["predicted_class_idx"],
        device="cpu",
    )

    # Run on CUDA
    cuda_res = WaferInferenceService.predict(
        model=cuda_model, input_tensor=tensor_input, device="cuda"
    )
    cuda_cam = WaferExplainabilityService.generate_explanation(
        model=cuda_model,
        input_tensor=tensor_input,
        wafer_map_2d=prep_2d,
        target_class=cuda_res["predicted_class_idx"],
        device="cuda",
    )

    # Assert predictions are identical
    assert cpu_res["predicted_class"] == cuda_res["predicted_class"]
    assert abs(cpu_res["confidence"] - cuda_res["confidence"]) < 1e-4

    # Assert Grad-CAM heatmaps are numerically equivalent within float32 tolerance
    max_cam_diff = float(np.max(np.abs(cpu_cam["heatmap"] - cuda_cam["heatmap"])))
    assert max_cam_diff < 1e-3, f"CPU vs CUDA Grad-CAM heatmap deviation: {max_cam_diff}"


# =============================================================================
# 7. Central Configuration & Startup Health Check Integration Tests
# =============================================================================


def test_startup_healthcheck_success():
    """Verify that startup health check passes on default system configuration."""
    from app.config import get_app_config, run_startup_healthcheck

    config = get_app_config()
    report = run_startup_healthcheck(config)
    assert report["status"] == "healthy"
    assert report["total_parameters"] == 520681
    assert report["trainable_parameters"] == 0
    assert report["num_classes"] == 9
    assert report["input_shape"] == [1, 1, 64, 64]


def test_startup_healthcheck_device_override():
    """Verify device override behavior in health check."""
    from app.config import AppConfig, run_startup_healthcheck

    cpu_config = AppConfig(device_setting="cpu")
    report = run_startup_healthcheck(cpu_config)
    assert report["device"] == "cpu"
    assert report["status"] == "healthy"


def test_startup_healthcheck_missing_checkpoint():
    """Verify health check raises CheckpointNotFoundError when checkpoint is missing."""
    from app.config import AppConfig, CheckpointNotFoundError, run_startup_healthcheck

    bad_config = AppConfig(checkpoint_path=Path("nonexistent_model.pt"))
    with pytest.raises(CheckpointNotFoundError, match="Checkpoint not found at"):
        run_startup_healthcheck(bad_config)


def test_startup_healthcheck_checksum_mismatch(tmp_path):
    """Verify health check raises ModelIntegrityError on SHA-256 hash mismatch."""
    from app.config import AppConfig, ModelIntegrityError, run_startup_healthcheck

    fake_ckpt = tmp_path / "fake_model.pt"
    fake_ckpt.write_bytes(b"dummy fake weights")

    bad_config = AppConfig(checkpoint_path=fake_ckpt)
    with pytest.raises(ModelIntegrityError, match="Cryptographic integrity mismatch"):
        run_startup_healthcheck(bad_config)
