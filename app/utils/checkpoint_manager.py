"""Cryptographic checkpoint verification and atomic asset downloader for WaferVision AI."""

import hashlib
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_CHECKPOINT_SHA256 = "b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "exp1_custom_cnn_baseline_best.pt"
)
DEFAULT_RELEASE_TAG = "v1.0.0"
DEFAULT_ASSET_NAME = "exp1_custom_cnn_baseline_best.pt"
DEFAULT_CHECKPOINT_URL = (
    f"https://github.com/darkPegsus/Semiconductor-wafer-defect-detection/releases/download/"
    f"{DEFAULT_RELEASE_TAG}/{DEFAULT_ASSET_NAME}"
)


class CheckpointDownloadError(RuntimeError):
    """Raised when remote checkpoint download fails."""


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        path: Path to target file.

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
    """Verify that checkpoint file exists and matches the expected cryptographic SHA-256 digest.

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


def download_checkpoint_atomically(
    download_url: str,
    target_path: Path | str,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download checkpoint from public URL with atomic write and SHA-256 pre-verification.

    The file is written to a temporary staging path in the target directory, its SHA-256
    is verified before moving into the final destination, and corrupted or incomplete downloads
    are strictly discarded and never moved to the target path.

    Args:
        download_url: Public direct download URL.
        target_path: Destination path for the verified checkpoint.
        expected_sha256: Expected SHA-256 checksum.
        progress_callback: Optional callback(downloaded_bytes, total_bytes).

    Returns:
        Destination Path of the verified checkpoint.

    Raises:
        CheckpointDownloadError: If download fails or returns non-200.
        ValueError: If downloaded file fails SHA-256 verification.
    """
    dest = Path(target_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use unique temporary file in the same filesystem directory to guarantee atomic rename
    tmp_path = dest.parent / f"{dest.name}.tmp.{uuid.uuid4().hex[:8]}"

    logger.info(f"Initiating atomic download from: {download_url}")
    logger.info(f"Staging to temporary location: {tmp_path.name}")

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "WaferVisionAI/1.0 (Streamlit Cloud Deployment)"},
    )

    hasher = hashlib.sha256()
    downloaded_bytes = 0

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            content_length = resp.headers.get("Content-Length")
            total_bytes = (
                int(content_length) if content_length and content_length.isdigit() else None
            )

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
                    downloaded_bytes += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded_bytes, total_bytes)

    except urllib.error.HTTPError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise CheckpointDownloadError(
            f"Failed to download checkpoint (HTTP {e.code}: {e.reason}) from:\n{download_url}\n\n"
            "Diagnostic Notice:\n"
            "If deploying to Streamlit Community Cloud, verify that Release 'v1.0.0' exists on GitHub "
            "and that asset 'exp1_custom_cnn_baseline_best.pt' is attached and publicly downloadable.\n"
            "Alternatively, set the environment variable WAFER_CHECKPOINT_URL to a valid direct download mirror."
        ) from e
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise CheckpointDownloadError(
            f"Network error downloading checkpoint from {download_url}: {e}"
        ) from e

    # Cryptographic integrity check prior to moving
    actual_hash = hasher.hexdigest()
    if actual_hash != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded checkpoint integrity mismatch!\n"
            f"Expected SHA-256: {expected_sha256}\n"
            f"Actual SHA-256:   {actual_hash}\n"
            "The temporary file was safely deleted and will not be cached."
        )

    # Atomic move to final cache location
    os.replace(tmp_path, dest)
    logger.info(
        f"Checkpoint successfully verified and cached at: {dest} ({downloaded_bytes:,} bytes, SHA-256: {actual_hash[:16]}...)"
    )
    return dest


def ensure_checkpoint_available(
    checkpoint_path: Path | str | None = None,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    download_url: str | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Ensure that the model checkpoint is present and verified on disk, downloading if missing.

    Resolution order:
      1. If file exists at checkpoint_path and SHA-256 matches: use existing file.
      2. If file exists but is corrupted: log warning, delete corrupted file, attempt re-download.
      3. If file does not exist: download atomically from download_url or WAFER_CHECKPOINT_URL.

    Args:
        checkpoint_path: Target path on disk. If None, uses DEFAULT_CHECKPOINT_PATH.
        expected_sha256: Expected hexadecimal hash string.
        download_url: Optional explicit download URL. If None, checks env var WAFER_CHECKPOINT_URL
                      or defaults to DEFAULT_CHECKPOINT_URL.
        progress_callback: Optional callback for download progress.

    Returns:
        Verified Path to the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint is missing and download is disabled/failed.
        ValueError: If checkpoint fails SHA-256 verification.
    """
    target = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH).resolve()

    # 1. Existing local file verification
    if target.exists():
        actual_hash = compute_file_sha256(target)
        if actual_hash == expected_sha256:
            logger.info(f"Local checkpoint verified: {target.name}")
            return target
        else:
            logger.warning(
                f"Existing checkpoint at {target} failed SHA-256 verification. Discarding corrupted file."
            )
            target.unlink(missing_ok=True)

    # 2. Resolve download URL
    url = download_url or os.getenv("WAFER_CHECKPOINT_URL") or DEFAULT_CHECKPOINT_URL

    logger.info(f"Checkpoint not present locally. Attempting retrieval from: {url}")
    return download_checkpoint_atomically(
        download_url=url,
        target_path=target,
        expected_sha256=expected_sha256,
        progress_callback=progress_callback,
    )
