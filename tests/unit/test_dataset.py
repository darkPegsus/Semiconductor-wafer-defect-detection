"""Unit tests for dataset loading, scalar extraction, and PyTorch abstractions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WM811kDataset,
    load_labeled_dataset,
)
from src.data.validator import extract_scalar_label


def test_extract_scalar_label():
    """Verify scalar string extraction handles varied nesting structures."""
    assert extract_scalar_label(np.array([["Center"]])) == "Center"
    assert extract_scalar_label([["none"]]) == "none"
    assert extract_scalar_label(["Scratch"]) == "Scratch"
    assert extract_scalar_label("Donut") == "Donut"
    assert extract_scalar_label([]) == "unlabeled"
    assert extract_scalar_label(np.array([])) == "unlabeled"
    assert extract_scalar_label([[]]) == "unlabeled"
    assert extract_scalar_label(float("nan")) == "unlabeled"


def test_label_mappings():
    """Verify bidirectional 9-class label-to-index mappings."""
    expected_classes = {
        "none",
        "Center",
        "Donut",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "Random",
        "Scratch",
        "Near-full",
    }
    assert set(LABEL_TO_IDX.keys()) == expected_classes
    assert set(IDX_TO_LABEL.values()) == expected_classes
    assert len(LABEL_TO_IDX) == 9
    for name, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == name


@pytest.mark.skipif(
    not Path("data/interim/labeled_wafers.pkl").exists(),
    reason="Interim labeled dataset not yet created",
)
def test_interim_dataset_integrity():
    """Validate 172,950 records, columns, types, and die values."""
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")

    assert len(df) == 172950, f"Expected 172,950 records, got {len(df)}"
    expected_columns = {
        "waferMap",
        "dieSize",
        "lotName",
        "waferIndex",
        "trianTestLabel",
        "failureType",
        "originalSplit",
    }
    assert expected_columns.issubset(set(df.columns))

    # All labels must be valid classes
    assert set(df["failureType"].unique()).issubset(set(LABEL_TO_IDX.keys()))
    assert "unlabeled" not in df["failureType"].unique()

    # Original split sanity
    assert set(df["originalSplit"].unique()) == {"Training", "Test"}

    # Wafer map integrity on first 100 samples
    for wm in df["waferMap"].iloc[:100]:
        assert isinstance(wm, np.ndarray)
        assert wm.ndim == 2
        assert wm.dtype == np.uint8
        unique_vals = set(np.unique(wm))
        assert unique_vals.issubset({0, 1, 2})


def test_wm811k_pytorch_dataset():
    """Verify PyTorch Dataset item loading, tensor shape, and types."""
    sample_data = pd.DataFrame(
        {
            "waferMap": [np.zeros((30, 30), dtype=np.uint8)],
            "failureType": ["Center"],
            "lotName": ["lot_test_1"],
            "waferIndex": [5],
        }
    )

    # 1. Default dataset produces canonical 64x64 single-channel tensor
    dataset = WM811kDataset(sample_data)
    assert len(dataset) == 1

    item = dataset[0]
    assert "image" in item
    assert "label" in item
    assert "label_name" in item
    assert "lot_name" in item
    assert "wafer_index" in item

    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (1, 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["label"].item() == LABEL_TO_IDX["Center"]
    assert item["label_name"] == "Center"
    assert item["lot_name"] == "lot_test_1"
    assert item["wafer_index"] == 5

    # 2. Configurable 3-channel one-hot dataset produces (3, 64, 64)
    dataset_one_hot = WM811kDataset(sample_data, representation="one_hot")
    item_one_hot = dataset_one_hot[0]
    assert item_one_hot["image"].shape == (3, 64, 64)

    # 3. Custom transform preserving raw native resolution (1, 30, 30)
    dataset_custom = WM811kDataset(
        sample_data,
        transform=lambda wm: torch.from_numpy(wm).unsqueeze(0).float(),
    )
    item_custom = dataset_custom[0]
    assert item_custom["image"].shape == (1, 30, 30)
