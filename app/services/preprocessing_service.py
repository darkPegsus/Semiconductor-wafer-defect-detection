"""Wafer map input validation and preprocessing service."""

import io
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.data.preprocessor import preprocess_letterbox_padding
from src.data.transforms import encode_single_channel

# Known categorical color palettes for wafer defect map rendering:
# Format: 3 RGB centroids for [0: exterior background, 1: normal die, 2: defective die]
KNOWN_CATEGORICAL_PALETTES: dict[str, dict[str, Any]] = {
    "viridis": {
        "name": "Viridis (WM-811K / Scientific)",
        "centroids": np.array(
            [
                [68, 1, 84],  # 0: Purple / Deep Violet (exterior background)
                [32, 144, 140],  # 1: Teal / Cyan (normal die)
                [253, 231, 36],  # 2: Yellow (defective die)
            ],
            dtype=np.float32,
        ),
        "description": "Purple (exterior), Teal (normal), Yellow (defect)",
    },
    "red_gray_dark": {
        "name": "Red-Gray-Dark (WaferVision Standard)",
        "centroids": np.array(
            [
                [28, 32, 42],  # 0: Dark Charcoal (exterior background)
                [170, 175, 185],  # 1: Slate Gray (normal die)
                [235, 45, 45],  # 2: Crimson Red (defective die)
            ],
            dtype=np.float32,
        ),
        "description": "Dark Charcoal (exterior), Slate Gray (normal), Crimson (defect)",
    },
    "high_contrast_green_red": {
        "name": "High-Contrast Green/Red/Black",
        "centroids": np.array(
            [
                [0, 0, 0],  # 0: Black (exterior background)
                [0, 200, 0],  # 1: Green (normal die)
                [200, 0, 0],  # 2: Red (defective die)
            ],
            dtype=np.float32,
        ),
        "description": "Black (exterior), Green (normal), Red (defect)",
    },
    "white_background_red_gray": {
        "name": "White-Background Red/Gray",
        "centroids": np.array(
            [
                [255, 255, 255],  # 0: White (exterior background)
                [170, 175, 185],  # 1: Slate Gray (normal die)
                [235, 45, 45],  # 2: Crimson Red (defective die)
            ],
            dtype=np.float32,
        ),
        "description": "White (exterior), Slate Gray (normal), Crimson (defect)",
    },
}


class WaferPreprocessingService:
    """Service to validate, normalize, and preprocess uploaded wafer maps."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"}
    MIN_DIMENSION = 8
    MAX_DIMENSION = 2048
    COLOR_TOLERANCE_TAU = 75.0
    MAX_UNMATCHED_RATIO = 0.15

    @classmethod
    def decode_categorical_rgb(
        cls,
        rgb_arr: np.ndarray,
        tau: float = COLOR_TOLERANCE_TAU,
        max_unmatched_ratio: float = MAX_UNMATCHED_RATIO,
    ) -> tuple[np.ndarray, str, float]:
        """Decode an RGB image into discrete wafer die categories {0, 1, 2}.

        Compares pixel colors against known wafer map palettes (Viridis, Red-Gray-Dark,
        High-Contrast) via Euclidean distance to palette centroids.

        Args:
            rgb_arr: uint8 numpy array of shape (H, W, 3).
            tau: Maximum Euclidean RGB distance threshold for a pixel to match a centroid.
            max_unmatched_ratio: Maximum tolerated fraction of unmatched pixels before rejecting.

        Returns:
            Tuple of:
            - decoded_map: 2D uint8 numpy array with values in {0, 1, 2}.
            - palette_name: Key of the matched palette.
            - unmatched_ratio: Fraction of pixels exceeding distance tau.

        Raises:
            ValueError: If image does not match any recognized semiconductor wafer palette.
        """
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB array, got shape {rgb_arr.shape}")

        rgb_float = rgb_arr.astype(np.float32)

        best_palette_key = None
        best_mean_dist = float("inf")
        best_dists = None

        for key, pinfo in KNOWN_CATEGORICAL_PALETTES.items():
            centroids = pinfo["centroids"]
            diffs = rgb_float[:, :, None, :] - centroids[None, None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            min_dist = np.min(dists, axis=-1)
            mean_dist = float(np.mean(min_dist))

            if mean_dist < best_mean_dist:
                best_mean_dist = mean_dist
                best_palette_key = key
                best_dists = dists

        if best_dists is None or best_palette_key is None:
            raise ValueError("No valid color palette could be evaluated.")

        min_dists = np.min(best_dists, axis=-1)
        unmatched_ratio = float(np.mean(min_dists > tau))

        if unmatched_ratio > max_unmatched_ratio:
            supported = ", ".join(pinfo["name"] for pinfo in KNOWN_CATEGORICAL_PALETTES.values())
            raise ValueError(
                f"Uploaded image does not match any recognized semiconductor wafer color palette. "
                f"{unmatched_ratio * 100:.1f}% of pixels exceed color tolerance (tau={tau}). "
                f"Recognized palettes: {supported}."
            )

        decoded_map = np.argmin(best_dists, axis=-1).astype(np.uint8)
        return decoded_map, best_palette_key, unmatched_ratio

    @classmethod
    def reconstruct_die_grid(
        cls,
        pixel_map: np.ndarray,
        target_grid: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Reconstruct discrete wafer die cells from a pixel-level categorical map.

        High-resolution wafer images (e.g. 640x640) render each silicon die across a block
        of multiple pixels. This method partitions the pixel map into discrete die grid cells
        and computes the categorical mode (majority vote) for each cell without interpolation.

        Args:
            pixel_map: 2D uint8 array with categorical die values {0, 1, 2}.
            target_grid: Optional target grid dimensions (gh, gw). If None, inferred automatically.

        Returns:
            2D uint8 numpy array with discrete values {0, 1, 2} of shape target_grid.
        """
        if pixel_map.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {pixel_map.shape}")

        h, w = pixel_map.shape

        if target_grid is None:
            # If native or near-native resolution, keep original die cells
            factor = max(1, int(round(max(h, w) / 64.0)))
            if factor <= 1 or max(h, w) <= 100:
                return pixel_map.astype(np.uint8)

            # Invert integer downsampling factor (e.g. 640x640 -> factor 10 -> 64x64)
            gh = max(8, int(round(h / factor)))
            gw = max(8, int(round(w / factor)))
            target_grid = (gh, gw)

        gh, gw = target_grid

        # When target grid matches pixel map shape exactly, return directly
        if (gh, gw) == (h, w):
            return pixel_map.astype(np.uint8)

        # Block majority voting
        row_bounds = np.round(np.linspace(0, h, gh + 1)).astype(int)
        col_bounds = np.round(np.linspace(0, w, gw + 1)).astype(int)

        reconstructed = np.zeros((gh, gw), dtype=np.uint8)
        for r in range(gh):
            r0, r1 = row_bounds[r], row_bounds[r + 1]
            for c in range(gw):
                c0, c1 = col_bounds[c], col_bounds[c + 1]
                patch = pixel_map[r0:r1, c0:c1]
                if patch.size == 0:
                    continue
                counts = np.bincount(patch.ravel(), minlength=3)
                reconstructed[r, c] = np.argmax(counts)

        return reconstructed

    @classmethod
    def load_wafer_from_bytes(
        cls,
        file_bytes: bytes | BinaryIO,
        filename: str = "",
        target_grid: tuple[int, int] | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Load and parse wafer map from raw byte stream or uploaded file.

        Supports image formats (PNG, JPG, etc.) and NumPy serialized files (.npy).

        Args:
            file_bytes: Raw bytes or file-like object.
            filename: Original filename to check extension.
            target_grid: Optional explicit die grid dimensions.
            return_metadata: Whether to return decoding metadata alongside the wafer array.

        Returns:
            2D numpy array of integer die values, or tuple of (array, metadata).

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
                wafer_map = cls.standardize_raw_wafer_matrix(arr, target_grid=target_grid)
                if return_metadata:
                    meta = {
                        "input_mode": "NPY",
                        "original_dimensions": list(arr.shape),
                        "reconstructed_grid": list(wafer_map.shape),
                        "detected_palette": "numpy_direct",
                        "palette_name": "Direct NumPy Array",
                        "unmatched_ratio": 0.0,
                    }
                    return wafer_map, meta
                return wafer_map
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

        return cls.pil_to_wafer_map(
            pil_img, target_grid=target_grid, return_metadata=return_metadata
        )

    @classmethod
    def pil_to_wafer_map(
        cls,
        pil_img: Image.Image,
        target_grid: tuple[int, int] | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Convert a PIL Image into a 2D discrete wafer map array.

        Handles:
        1. RGB/RGBA wafer maps with categorical palettes (Viridis, Red-Gray-Dark, etc.).
        2. Grayscale wafer maps with discrete levels {0, 1, 2} or standard 8-bit range.
        3. Multi-pixel die grid reconstruction via categorical majority voting.

        Args:
            pil_img: PIL Image object.
            target_grid: Optional explicit die grid dimensions (gh, gw).
            return_metadata: Whether to return decoding metadata alongside the wafer array.

        Returns:
            2D uint8 numpy array with discrete values {0: exterior, 1: normal, 2: defect},
            or (array, metadata) if return_metadata=True.
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

        metadata: dict[str, Any] = {
            "input_mode": pil_img.mode,
            "original_dimensions": [w, h],
        }

        # RGB / RGBA Image parsing
        if pil_img.mode in ("RGB", "RGBA", "P"):
            if pil_img.mode == "RGBA":
                alpha = np.array(pil_img.getchannel("A"), dtype=np.uint8)
                transparent_mask = alpha < 64
                rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            else:
                transparent_mask = np.zeros((h, w), dtype=bool)
                rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

            decoded_pixel_map, palette_key, unmatched_ratio = cls.decode_categorical_rgb(
                rgb, tau=cls.COLOR_TOLERANCE_TAU, max_unmatched_ratio=cls.MAX_UNMATCHED_RATIO
            )

            if transparent_mask.any():
                decoded_pixel_map[transparent_mask] = 0

            reconstructed_map = cls.reconstruct_die_grid(decoded_pixel_map, target_grid=target_grid)

            metadata["detected_palette"] = palette_key
            metadata["palette_name"] = KNOWN_CATEGORICAL_PALETTES[palette_key]["name"]
            metadata["unmatched_ratio"] = round(unmatched_ratio, 4)
            metadata["reconstructed_grid"] = list(reconstructed_map.shape)

            if return_metadata:
                return reconstructed_map, metadata
            return reconstructed_map

        # Grayscale / Single-channel Image parsing
        gray = np.array(pil_img.convert("L"), dtype=np.uint8)
        unique_vals = np.unique(gray)

        # If image already contains discrete values {0, 1, 2}
        if set(unique_vals).issubset({0, 1, 2}):
            pixel_map = gray
            palette_desc = "Discrete Grayscale {0, 1, 2}"
        else:
            # Standard quantization (e.g. 0=exterior, 127/128=normal, 255=defect)
            pixel_map = np.zeros_like(gray, dtype=np.uint8)
            pixel_map[(gray >= 50) & (gray < 200)] = 1
            pixel_map[gray >= 200] = 2
            palette_desc = "Quantized Grayscale (0, 128, 255)"

        reconstructed_map = cls.reconstruct_die_grid(pixel_map, target_grid=target_grid)

        metadata["detected_palette"] = "grayscale"
        metadata["palette_name"] = palette_desc
        metadata["unmatched_ratio"] = 0.0
        metadata["reconstructed_grid"] = list(reconstructed_map.shape)

        if return_metadata:
            return reconstructed_map, metadata
        return reconstructed_map

    @classmethod
    def standardize_raw_wafer_matrix(
        cls,
        arr: np.ndarray,
        target_grid: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Validate and standardize a raw 2D NumPy array.

        Args:
            arr: 2D array.
            target_grid: Optional target die grid dimensions.

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
            norm_arr = np.zeros_like(int_arr)
            norm_arr[int_arr == 1] = 1
            norm_arr[int_arr >= 2] = 2
            int_arr = norm_arr

        return cls.reconstruct_die_grid(int_arr, target_grid=target_grid)

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
