"""Unit tests for checkpoint manager: atomic download, SHA-256 verification, and caching."""

import hashlib
import io
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from app.utils.checkpoint_manager import (
    CheckpointDownloadError,
    compute_file_sha256,
    download_checkpoint_atomically,
    ensure_checkpoint_available,
    verify_checkpoint_integrity,
)


@pytest.fixture
def dummy_checkpoint_data():
    """Deterministic synthetic checkpoint data and its SHA-256 digest."""
    data = b"STREAMLIT_COMMUNITY_CLOUD_VERIFIED_CHECKPOINT_PAYLOAD_V1" * 1024
    digest = hashlib.sha256(data).hexdigest()
    return data, digest


def _create_mock_response(data: bytes, status: int = 200, headers: dict | None = None):
    """Create a mock HTTP response object compatible with urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.headers = headers or {"Content-Length": str(len(data))}
    # Make chunked reads work
    stream = io.BytesIO(data)
    mock_resp.read = stream.read
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def test_compute_file_sha256(tmp_path):
    """Verify SHA-256 computation matches known digest."""
    test_file = tmp_path / "sample.bin"
    content = b"WaferVisionAI_Deterministic_Test_String"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(test_file) == expected

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "nonexistent.bin")


def test_verify_checkpoint_integrity(tmp_path):
    """Verify integrity validation passes on match and fails on mismatch."""
    valid_file = tmp_path / "valid.pt"
    content = b"VALID_MODEL_BYTES"
    valid_file.write_bytes(content)
    valid_hash = hashlib.sha256(content).hexdigest()

    assert verify_checkpoint_integrity(valid_file, valid_hash) is True

    # Bad hash
    with pytest.raises(ValueError, match="integrity check failed"):
        verify_checkpoint_integrity(
            valid_file, "0000000000000000000000000000000000000000000000000000000000000000"
        )

    # Missing file
    with pytest.raises(FileNotFoundError):
        verify_checkpoint_integrity(tmp_path / "missing.pt", valid_hash)


def test_cold_start_download_and_atomic_placement(dummy_checkpoint_data, tmp_path):
    """Test Requirement 9: Cold-start download verifies hash and moves atomically."""
    data, digest = dummy_checkpoint_data
    dest_path = tmp_path / "models" / "checkpoint.pt"

    assert not dest_path.exists()

    mock_resp = _create_mock_response(data)
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        downloaded = download_checkpoint_atomically(
            download_url="https://example.com/models/exp1_custom_cnn_baseline_best.pt",
            target_path=dest_path,
            expected_sha256=digest,
        )
        mock_urlopen.assert_called_once()

    assert downloaded.exists()
    assert downloaded.resolve() == dest_path.resolve()
    assert dest_path.stat().st_size == len(data)
    assert compute_file_sha256(dest_path) == digest

    # Verify no temporary files remain in directory
    tmp_files = list(dest_path.parent.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_subsequent_cached_startup(dummy_checkpoint_data, tmp_path):
    """Test Requirement 9: Cached startup does not re-download if file exists and hash matches."""
    data, digest = dummy_checkpoint_data
    dest_path = tmp_path / "cached_model.pt"
    dest_path.write_bytes(data)

    with patch("urllib.request.urlopen") as mock_urlopen:
        cached_path = ensure_checkpoint_available(
            checkpoint_path=dest_path,
            expected_sha256=digest,
            download_url="https://example.com/should_never_be_called.pt",
        )
        mock_urlopen.assert_not_called()

    assert cached_path == dest_path
    assert cached_path.exists()


def test_corrupted_download_rejection(dummy_checkpoint_data, tmp_path):
    """Test Requirement 3: Corrupted downloads are rejected, never moved to destination."""
    data, digest = dummy_checkpoint_data
    dest_path = tmp_path / "target_never_created.pt"

    bad_data = b"CORRUPTED_BYTES" + data[15:]
    mock_resp = _create_mock_response(bad_data)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(ValueError, match="integrity mismatch"):
            download_checkpoint_atomically(
                download_url="https://example.com/corrupted.pt",
                target_path=dest_path,
                expected_sha256=digest,
            )

    # Assert destination was NEVER created
    assert not dest_path.exists()

    # Assert temporary file was cleaned up
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_http_404_handling(tmp_path):
    """Test Requirement 1 & 8: HTTP 404 produces clear diagnostic error."""
    dest_path = tmp_path / "missing.pt"
    http_error = urllib.error.HTTPError(
        url="https://example.com/nonexistent.pt",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b"Not Found"),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(CheckpointDownloadError, match="HTTP 404"):
            download_checkpoint_atomically(
                download_url="https://example.com/nonexistent.pt",
                target_path=dest_path,
                expected_sha256="dummy",
            )

    assert not dest_path.exists()


def test_corrupted_existing_cache_handling(dummy_checkpoint_data, tmp_path):
    """Ensure corrupted existing cache is deleted and re-downloaded."""
    data, digest = dummy_checkpoint_data
    dest_path = tmp_path / "corrupted_cache.pt"
    dest_path.write_bytes(b"CORRUPTED_CACHE_BYTES")

    mock_resp = _create_mock_response(data)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ensure_checkpoint_available(
            checkpoint_path=dest_path,
            expected_sha256=digest,
            download_url="https://example.com/valid.pt",
        )

    assert result.exists()
    assert compute_file_sha256(result) == digest
