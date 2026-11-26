"""Central application configuration and healthcheck service for WaferVision AI."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

# -----------------------------------------------------------------------------
# Base Project Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Checkpoint and Architecture Invariant Constants
# -----------------------------------------------------------------------------
EXPECTED_CHECKPOINT_SHA256 = "b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "exp1_custom_cnn_baseline_best.pt"
)

CANONICAL_INPUT_SHAPE = (1, 1, 64, 64)
NUM_CLASSES = 9
EXPECTED_TOTAL_PARAMETERS = 520_681
TARGET_GRAD_CAM_LAYER = "features.conv_block4"

# -----------------------------------------------------------------------------
# Input Validation Constraints
# -----------------------------------------------------------------------------
MIN_WAFER_DIMENSION = 8
MAX_WAFER_DIMENSION = 2048
ALLOWED_FILE_EXTENSIONS = (".npy", ".png", ".jpg", ".jpeg", ".bmp")
SUPPORTED_PALETTES = (
    "viridis",
    "red_gray_dark",
    "high_contrast_green_red",
    "white_background_red_gray",
)

DEFECT_CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]


class ConfigurationError(Exception):
    """Raised when application configuration is invalid."""


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the specified model checkpoint file does not exist."""


class ModelIntegrityError(ValueError):
    """Raised when the model checkpoint fails cryptographic or architectural validation."""


class HealthCheckError(RuntimeError):
    """Raised when startup health check fails."""


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for WaferVision AI."""

    project_root: Path = PROJECT_ROOT
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256
    device_setting: str = field(
        default_factory=lambda: os.getenv("WAFER_DEVICE", "auto").strip().lower()
    )
    target_layer: str = TARGET_GRAD_CAM_LAYER
    verify_checksum_on_load: bool = True

    @property
    def resolved_device(self) -> torch.device:
        """Resolve device string to torch.device with fallback logic."""
        if self.device_setting in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested via WAFER_DEVICE but not available; falling back to CPU."
                )
                return torch.device("cpu")
            return torch.device("cuda")
        elif self.device_setting == "cpu":
            return torch.device("cpu")
        elif self.device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            logger.warning(
                f"Unknown WAFER_DEVICE='{self.device_setting}'; falling back to auto-detection."
            )
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_app_config() -> AppConfig:
    """Factory to produce an AppConfig instance with environment variable overrides."""
    ckpt_env = os.getenv("WAFER_CHECKPOINT_PATH")
    if ckpt_env:
        resolved_ckpt = Path(ckpt_env)
        if not resolved_ckpt.is_absolute():
            resolved_ckpt = (PROJECT_ROOT / resolved_ckpt).resolve()
    else:
        resolved_ckpt = DEFAULT_CHECKPOINT_PATH

    device_env = os.getenv("WAFER_DEVICE", "auto").strip().lower()

    return AppConfig(
        project_root=PROJECT_ROOT,
        checkpoint_path=resolved_ckpt,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
        device_setting=device_env,
        target_layer=TARGET_GRAD_CAM_LAYER,
        verify_checksum_on_load=True,
    )


def run_startup_healthcheck(config: AppConfig | None = None) -> dict[str, Any]:
    """Execute a comprehensive startup health check to verify system integrity.

    Verifies:
      1. Checkpoint file existence and absolute path resolution.
      2. Cryptographic SHA-256 integrity match.
      3. Model architecture instantiation and parameter freezing (requires_grad = False).
      4. Single forward pass with canonical input tensor [1, 1, 64, 64].
      5. Output logits dimensions [1, 9] and finite probability values.
      6. Grad-CAM hook resolution on target layer.

    Args:
        config: Optional AppConfig. If None, loaded via get_app_config().

    Returns:
        Structured health report dictionary.

    Raises:
        CheckpointNotFoundError: If checkpoint path does not exist.
        ModelIntegrityError: If checksum or parameter counts mismatch.
        HealthCheckError: If forward pass or Grad-CAM hook execution fails.
    """
    if config is None:
        config = get_app_config()

    logger.info("Running WaferVision AI startup health check...")

    # 1. Checkpoint existence and automated retrieval
    if not config.checkpoint_path.exists():
        from app.utils.checkpoint_manager import ensure_checkpoint_available

        try:
            ensure_checkpoint_available(
                checkpoint_path=config.checkpoint_path,
                expected_sha256=config.expected_sha256,
            )
        except Exception as e:
            raise CheckpointNotFoundError(
                f"Checkpoint not found at: {config.checkpoint_path.resolve()}\n"
                f"Automated retrieval failed: {e}\n"
                "Ensure the model checkpoint file is placed in models/checkpoints/ or set WAFER_CHECKPOINT_PATH or WAFER_CHECKPOINT_URL."
            ) from e

    # 2. Checksum verification
    from app.utils.model_loader import compute_file_sha256

    actual_hash = compute_file_sha256(config.checkpoint_path)
    if actual_hash != config.expected_sha256:
        raise ModelIntegrityError(
            f"Cryptographic integrity mismatch for checkpoint: {config.checkpoint_path}\n"
            f"Expected SHA-256: {config.expected_sha256}\n"
            f"Actual SHA-256:   {actual_hash}"
        )

    # 3. Model loading & parameter check
    target_device = config.resolved_device
    from app.utils.model_loader import load_frozen_model

    model, metadata, actual_device = load_frozen_model(
        checkpoint_path=config.checkpoint_path,
        device=target_device,
        verify_hash=False,  # Already verified above
    )

    if metadata["trainable_parameters"] != 0:
        raise ModelIntegrityError(
            f"Model is not frozen! Found {metadata['trainable_parameters']} trainable parameters."
        )
    if metadata["total_parameters"] != EXPECTED_TOTAL_PARAMETERS:
        raise ModelIntegrityError(
            f"Parameter count mismatch! Expected {EXPECTED_TOTAL_PARAMETERS:,}, got {metadata['total_parameters']:,}."
        )

    # 4. Forward pass smoke test
    try:
        dummy_tensor = torch.zeros(CANONICAL_INPUT_SHAPE, dtype=torch.float32, device=actual_device)
        with torch.inference_mode():
            logits = model(dummy_tensor)
    except Exception as e:
        raise HealthCheckError(f"Model forward pass failed on {actual_device}: {e}") from e

    if logits.shape != (1, NUM_CLASSES):
        raise HealthCheckError(
            f"Expected logits shape (1, {NUM_CLASSES}), got {tuple(logits.shape)}"
        )
    if not torch.all(torch.isfinite(logits)):
        raise HealthCheckError("Forward pass produced non-finite (NaN or Inf) logits.")

    # 5. Grad-CAM hook smoke test
    try:
        from app.services.explainability_service import WaferExplainabilityService

        dummy_map_2d = np.zeros((64, 64), dtype=np.uint8)
        cam_result = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=dummy_tensor.cpu(),
            wafer_map_2d=dummy_map_2d,
            target_class=0,
            device=actual_device,
        )
    except Exception as e:
        raise HealthCheckError(f"Grad-CAM hook execution failed on {actual_device}: {e}") from e

    if cam_result["heatmap"].shape != (64, 64):
        raise HealthCheckError(
            f"Expected Grad-CAM heatmap shape (64, 64), got {cam_result['heatmap'].shape}"
        )

    health_report = {
        "status": "healthy",
        "device": str(actual_device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "total_parameters": metadata["total_parameters"],
        "trainable_parameters": metadata["trainable_parameters"],
        "model_architecture": "CustomWaferCNN",
        "input_shape": list(CANONICAL_INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "target_layer": config.target_layer,
    }

    logger.info(f"Startup health check PASSED on {actual_device}. Status: HEALTHY.")
    return health_report
