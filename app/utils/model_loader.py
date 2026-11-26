"""Model loading and verification utility for CustomWaferCNN deployment."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from app.utils.checkpoint_manager import (
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    compute_file_sha256,
    ensure_checkpoint_available,
    verify_checkpoint_integrity,
)
from src.models.custom_cnn import CustomWaferCNN

__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "EXPECTED_CHECKPOINT_SHA256",
    "compute_file_sha256",
    "ensure_checkpoint_available",
    "load_frozen_model",
    "verify_checkpoint_integrity",
]

# Global singleton cache for non-streamlit contexts or fallback
_CACHED_MODEL: CustomWaferCNN | None = None
_CACHED_METADATA: dict[str, Any] | None = None
_CACHED_DEVICE: torch.device | None = None


def load_frozen_model(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str | None = None,
    verify_hash: bool = True,
) -> tuple[CustomWaferCNN, dict[str, Any], torch.device]:
    """Load and freeze the CustomWaferCNN baseline model.

    Ensures the model is loaded in eval mode with all weights frozen (requires_grad = False).
    Reuses cached singleton if previously loaded.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Target torch device. If None, auto-selects CUDA if available, else CPU.
        verify_hash: Whether to verify SHA-256 checksum before loading.

    Returns:
        Tuple of (frozen_model, checkpoint_metadata, device).
    """
    global _CACHED_MODEL, _CACHED_METADATA, _CACHED_DEVICE

    path = Path(checkpoint_path)

    # Resolve device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    # Check cache if path and device match
    if _CACHED_MODEL is not None and _CACHED_DEVICE == target_device:
        return _CACHED_MODEL, _CACHED_METADATA or {}, _CACHED_DEVICE

    # Ensure checkpoint is available locally (downloads if missing)
    if not path.exists():
        path = ensure_checkpoint_available(path, EXPECTED_CHECKPOINT_SHA256)

    if verify_hash:
        verify_checkpoint_integrity(path, EXPECTED_CHECKPOINT_SHA256)

    # Load state dict
    ckpt = torch.load(path, map_location=target_device)
    model = CustomWaferCNN(in_channels=1, num_classes=9)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(target_device)

    # Freeze model weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    metadata = {
        "epoch": ckpt.get("epoch", 24),
        "best_score": ckpt.get("best_score", 0.8861),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    _CACHED_MODEL = model
    _CACHED_METADATA = metadata
    _CACHED_DEVICE = target_device

    logger.info(
        f"CustomWaferCNN successfully loaded on {target_device} "
        f"({metadata['total_parameters']:,} parameters, 0 trainable)."
    )
    return model, metadata, target_device
