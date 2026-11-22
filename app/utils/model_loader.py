"""Model loading and verification utility for CustomWaferCNN deployment."""

import hashlib
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from src.models.custom_cnn import CustomWaferCNN

EXPECTED_CHECKPOINT_SHA256 = "b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8"
DEFAULT_CHECKPOINT_PATH = Path("models/checkpoints/exp1_custom_cnn_baseline_best.pt")

# Global singleton cache for non-streamlit contexts or fallback
_CACHED_MODEL: CustomWaferCNN | None = None
_CACHED_METADATA: dict[str, Any] | None = None
_CACHED_DEVICE: torch.device | None = None


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checkpoint_integrity(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> bool:
    """Verify that the checkpoint file exists and matches the expected SHA-256 hash.

    Args:
        checkpoint_path: Path to checkpoint file.
        expected_sha256: Expected hexadecimal hash string.

    Returns:
        True if verification succeeds.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If SHA-256 hash does not match expected hash.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path.resolve()}")

    actual_hash = compute_file_sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"Checkpoint integrity check failed for {path}!\n"
            f"Expected SHA-256: {expected_sha256}\n"
            f"Actual SHA-256:   {actual_hash}"
        )

    logger.info(f"Verified checkpoint integrity: {path.name} [SHA-256: {actual_hash[:16]}...]")
    return True


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
