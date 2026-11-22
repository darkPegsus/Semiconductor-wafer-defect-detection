"""Wafer map input validation and preprocessing service."""

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.data.preprocessor import preprocess_letterbox_padding
from src.data.transforms import encode_single_channel


class WaferPreprocessingService:
    """Service to validate, normalize, and preprocess uploaded wafer maps."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"}
    MIN_DIMENSION = 8
    MAX_DIMENSION = 2048

    @classmethod
    def load_wafer_from_bytes(cls, file_bytes: bytes | BinaryIO, filename: str = "") -> np.ndarray:
        """Load and parse wafer map from raw byte stream or uploaded file.

        Supports image formats (PNG, JPG, etc.) and NumPy serialized files (.npy).

        Args:
            file_bytes: Raw bytes or file-like object.
            filename: Original filename to check extension.

        Returns:
            2D numpy array of integer die values.

        Raises:
            ValueError: If file is invalid, empty, or cannot be parsed.
        """
        if isinstance(file_bytes, bytes):
            stream = io.BytesIO(file_bytes)
        else:
            stream = file_bytes

        ext = Path(filename).suffix.lower()

        # Handle NumPy .npy uploads
        if ext == ".npy":
            try:
                stream.seek(0)
                arr = np.load(stream)
                return cls.standardize_raw_wafer_matrix(arr)
            except Exception as e:
                raise ValueError(f"Failed to parse NumPy array from .npy file: {e}") from e

        # Handle standard image uploads (PNG, JPG, etc.)
        try:
            stream.seek(0)
            pil_img = Image.open(stream)
            pil_img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(
                f"Unsupported or corrupted image file. Please upload a valid PNG, JPG, or NPY file. Details: {e}"
            ) from e

        return cls.pil_to_wafer_map(pil_img)

    @classmethod
    def pil_to_wafer_map(cls, pil_img: Image.Image) -> np.ndarray:
        """Convert a PIL Image into a 2D discrete wafer map array.

        Handles:
        1. RGB wafer maps with colored dies (e.g. Red defect, Gray normal, Dark exterior).
        2. Grayscale wafer maps with discrete levels (0, 1, 2) or standard 8-bit range (0-255).

        Args:
            pil_img: PIL Image object.

        Returns:
            2D uint8 numpy array with discrete values {0: exterior, 1: normal, 2: defect}.
        """
        w, h = pil_img.size
        if w < cls.MIN_DIMENSION or h < cls.MIN_DIMENSION:
            raise ValueError(
                f"Wafer map dimensions ({w}x{h}) are too small. Minimum dimension is {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION}."
            )
        if w > cls.MAX_DIMENSION or h > cls.MAX_DIMENSION:
            raise ValueError(
                f"Wafer map dimensions ({w}x{h}) exceed maximum allowed dimension ({cls.MAX_DIMENSION}px)."
            )

        # RGB Image parsing
        if pil_img.mode in ("RGB", "RGBA"):
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            r = rgb[:, :, 0].astype(int)
            g = rgb[:, :, 1].astype(int)
            b = rgb[:, :, 2].astype(int)

            wafer_map = np.zeros((h, w), dtype=np.uint8)

            # Heuristic for colored wafer maps:
            # Defect dies: High red component, significantly greater than green and blue
            defect_mask = (r > 150) & (r > g + 40) & (r > b + 40)

            # Exterior / background: Very dark pixels or very light background
            exterior_mask = (r < 50) & (g < 50) & (b < 50)

            # Normal dies: Neutral gray/slate (non-defect, non-exterior)
            normal_mask = ~defect_mask & ~exterior_mask

            wafer_map[normal_mask] = 1
            wafer_map[defect_mask] = 2
            return wafer_map

        # Grayscale / Single-channel Image parsing
        gray = np.array(pil_img.convert("L"), dtype=np.uint8)
        unique_vals = np.unique(gray)

        # If image already contains discrete values {0, 1, 2}
        if set(unique_vals).issubset({0, 1, 2}):
            return gray

        # If image uses standard quantization (e.g. 0=exterior, 127/128=normal, 255=defect)
        wafer_map = np.zeros_like(gray, dtype=np.uint8)
        wafer_map[(gray >= 50) & (gray < 200)] = 1
        wafer_map[gray >= 200] = 2
        return wafer_map

    @classmethod
    def standardize_raw_wafer_matrix(cls, arr: np.ndarray) -> np.ndarray:
        """Validate and standardize a raw 2D NumPy array.

        Args:
            arr: 2D array.

        Returns:
            2D uint8 numpy array with values in {0, 1, 2}.
        """
        if arr.ndim != 2:
            raise ValueError(
                f"Wafer map must be 2-dimensional, got {arr.ndim} dimensions (shape: {arr.shape})"
            )

        h, w = arr.shape
        if h < cls.MIN_DIMENSION or w < cls.MIN_DIMENSION:
            raise ValueError(
                f"Wafer map shape {arr.shape} is too small (minimum {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})"
            )

        int_arr = np.rint(arr).astype(np.uint8)
        unique_vals = set(np.unique(int_arr))

        if not unique_vals.issubset({0, 1, 2}):
            # Rescale / clip if arbitrary non-standard values
            norm_arr = np.zeros_like(int_arr)
            norm_arr[int_arr == 1] = 1
            norm_arr[int_arr >= 2] = 2
            return norm_arr

        return int_arr

    @classmethod
    def preprocess_for_inference(
        cls,
        wafer_map: np.ndarray,
        target_size: tuple[int, int] = (64, 64),
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Apply Method C letterbox preprocessing and convert to single-channel tensor.

        Method C preserves die aspect ratio through proportional resizing followed
        by symmetric zero-padding to target_size.

        Args:
            wafer_map: 2D discrete wafer array.
            target_size: Canonical model input dimensions (default 64x64).

        Returns:
            Tuple of:
            - preprocessed_map: 2D uint8 array of shape target_size.
            - tensor_batch: 4D FloatTensor of shape [1, 1, 64, 64].
        """
        # Step 1: Aspect-ratio preserving letterbox padding (Method C)
        preprocessed_2d = preprocess_letterbox_padding(
            wafer_map=wafer_map,
            target_size=target_size,
            pad_value=0,
        )

        # Step 2: Single-channel float encoding [1, H, W]
        tensor_1ch = encode_single_channel(preprocessed_2d)

        # Step 3: Add batch dimension -> [1, 1, 64, 64]
        tensor_batch = tensor_1ch.unsqueeze(0)

        return preprocessed_2d, tensor_batch
