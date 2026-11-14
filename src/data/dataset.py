"""Dataset abstraction and extraction utilities for WM-811K wafer defect detection."""

import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.core.indexes
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.data.transforms import WaferTransform
from src.data.validator import extract_scalar_label, load_raw_lswmd

# Compatibility patch for legacy Pandas pickles
if "pandas.indexes" not in sys.modules:
    sys.modules["pandas.indexes"] = pandas.core.indexes


def extract_labeled_dataset(
    raw_path: str | Path = "data/raw/LSWMD.pkl",
    output_path: str | Path = "data/interim/labeled_wafers.pkl",
) -> pd.DataFrame:
    """Extract, clean, and optimize all 172,950 labeled wafer records.

    Preserves waferMap, failureType, lotName, waferIndex, and original split.
    Converts labels to scalar strings and downcasts numeric types.

    Args:
        raw_path: Path to the raw LSWMD.pkl file.
        output_path: Destination path for the optimized pickle file.

    Returns:
        Cleaned pandas DataFrame containing only labeled records.
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw dataset from {raw_path}...")
    df = load_raw_lswmd(raw_path)

    logger.info("Extracting scalar failure types and split annotations...")
    df["failureType"] = df["failureType"].apply(extract_scalar_label)
    df["originalSplit"] = df["trianTestLabel"].apply(extract_scalar_label)

    # Filter strictly labeled records
    labeled_df = df[df["failureType"] != "unlabeled"].copy()
    num_labeled = len(labeled_df)
    logger.info(f"Filtered {num_labeled} labeled records from {len(df)} total records.")

    if num_labeled != 172950:
        raise ValueError(f"Expected exactly 172,950 labeled records, got {num_labeled}")

    # Downcast datatypes for memory efficiency
    labeled_df["waferIndex"] = labeled_df["waferIndex"].astype(np.int16)
    labeled_df["dieSize"] = labeled_df["dieSize"].astype(np.int32)
    labeled_df["lotName"] = labeled_df["lotName"].astype(str)
    labeled_df["failureType"] = labeled_df["failureType"].astype(str)
    labeled_df["originalSplit"] = labeled_df["originalSplit"].astype(str)

    # Ensure all wafer map matrices are uint8 and preserve die values [0, 1, 2]
    def _standardize_matrix(wm: np.ndarray) -> np.ndarray:
        if wm.dtype != np.uint8:
            return wm.astype(np.uint8)
        return wm

    labeled_df["waferMap"] = labeled_df["waferMap"].apply(_standardize_matrix)

    # Reset index for clean contiguous indexing
    labeled_df = labeled_df.reset_index(drop=True)

    logger.info(f"Saving optimized labeled dataset to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(labeled_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Successfully serialized {len(labeled_df)} records to {output_path}")
    return labeled_df


def load_labeled_dataset(path: str | Path = "data/interim/labeled_wafers.pkl") -> pd.DataFrame:
    """Load optimized labeled dataset from interim storage.

    Args:
        path: Path to the interim labeled wafers pickle file.

    Returns:
        DataFrame with labeled wafer records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Interim dataset not found at: {path}")

    with open(path, "rb") as f:
        df = pickle.load(f)

    return df


# Mapping from failureType string to integer class index
LABEL_TO_IDX: dict[str, int] = {
    "none": 0,
    "Center": 1,
    "Donut": 2,
    "Edge-Loc": 3,
    "Edge-Ring": 4,
    "Loc": 5,
    "Random": 6,
    "Scratch": 7,
    "Near-full": 8,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}


class WM811kDataset(Dataset):
    """PyTorch Dataset for WM-811K Wafer Maps."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Callable[[np.ndarray], torch.Tensor | np.ndarray] | None = None,
        preprocessing_method: str = "method_c",
        representation: str = "single_channel",
        target_size: tuple[int, int] = (64, 64),
        label_mapping: dict[str, int] | None = None,
    ) -> None:
        """Initialize wafer dataset.

        Args:
            dataframe: DataFrame containing 'waferMap' and 'failureType'.
            transform: Optional custom transform callable applied to the wafer map.
            preprocessing_method: Fallback preprocessing ('method_a', 'method_b', 'method_c').
            representation: Input representation ('single_channel' or 'one_hot').
            target_size: Target resolution tuple (H, W).
            label_mapping: Mapping from string class label to integer target.
        """
        self.df = dataframe.reset_index(drop=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = WaferTransform(
                preprocessing_method=preprocessing_method,  # type: ignore[arg-type]
                representation=representation,  # type: ignore[arg-type]
                target_size=target_size,
            )
        self.label_mapping = label_mapping or LABEL_TO_IDX

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        wafer_map = row["waferMap"]
        label_str = row["failureType"]
        label_idx = self.label_mapping[label_str]

        wafer_tensor = self.transform(wafer_map)
        if not isinstance(wafer_tensor, torch.Tensor):
            wafer_tensor = torch.from_numpy(wafer_tensor).float()

        return {
            "image": wafer_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_name": label_str,
            "lot_name": row["lotName"],
            "wafer_index": int(row["waferIndex"]),
        }
