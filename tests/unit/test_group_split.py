"""Unit tests for group-aware splitting and zero lot leakage verification."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.group_split import (
    compute_split_distributions,
    create_stratified_group_split,
    extract_original_benchmark_split,
    load_split_indices,
    verify_zero_lot_leakage,
)


@pytest.fixture
def synthetic_wafer_df() -> pd.DataFrame:
    """Create synthetic wafer dataset with 10 lots and 3 classes."""
    records = []
    classes = ["none", "Center", "Donut"]
    for lot_idx in range(10):
        lot_name = f"synth_lot_{lot_idx}"
        for w_idx in range(25):
            cls_name = classes[(lot_idx + w_idx) % len(classes)]
            records.append(
                {
                    "lotName": lot_name,
                    "failureType": cls_name,
                    "waferIndex": w_idx + 1,
                    "originalSplit": "Training" if lot_idx < 6 else "Test",
                }
            )
    return pd.DataFrame(records)


def test_verify_zero_lot_leakage_success(synthetic_wafer_df):
    """Verify clean disjoint splits pass the check."""
    train_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_0", "synth_lot_1"]))[0]
    val_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_2"]))[0]
    test_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_3", "synth_lot_4"]))[0]

    assert verify_zero_lot_leakage(synthetic_wafer_df, train_idx, val_idx, test_idx) is True


def test_verify_zero_lot_leakage_detects_leak(synthetic_wafer_df):
    """Verify error is raised if a lot is leaked across splits."""
    train_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_0", "synth_lot_1"]))[0]
    val_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_1", "synth_lot_2"]))[
        0
    ]  # Leak!
    test_idx = np.where(synthetic_wafer_df["lotName"].isin(["synth_lot_3"]))[0]

    with pytest.raises(AssertionError, match="Lot leakage detected"):
        verify_zero_lot_leakage(synthetic_wafer_df, train_idx, val_idx, test_idx)


def test_synthetic_group_split(synthetic_wafer_df):
    """Test group split creation on synthetic data."""
    splits = create_stratified_group_split(synthetic_wafer_df, random_state=42)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # Disjoint indices
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    assert len(set(splits["val"]) & set(splits["test"])) == 0

    # Total sample count preserved
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(
        synthetic_wafer_df
    )

    # Disjoint lots verified
    assert verify_zero_lot_leakage(
        synthetic_wafer_df, splits["train"], splits["val"], splits["test"]
    )


def test_extract_original_split(synthetic_wafer_df):
    """Test author benchmark extraction and lot verification."""
    splits = extract_original_benchmark_split(synthetic_wafer_df)
    assert "train" in splits
    assert "test" in splits
    assert len(set(splits["train"]) & set(splits["test"])) == 0


@pytest.mark.skipif(
    not Path("data/interim/split_indices.json").exists(),
    reason="Split indices file not yet generated",
)
def test_interim_split_indices_verification():
    """Verify saved split indices: 0 lot leakage and all rare classes present."""
    from src.data.dataset import load_labeled_dataset

    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # 1. Total samples check
    assert len(train_idx) + len(val_idx) + len(test_idx) == 172950

    # 2. Zero lot leakage check
    assert verify_zero_lot_leakage(df, train_idx, val_idx, test_idx) is True

    # 3. Class distributions: verify EVERY class (including Near-full) is present in all 3 splits
    dist = compute_split_distributions(df, splits)
    all_classes = set(df["failureType"].unique())
    assert len(all_classes) == 9

    for c in all_classes:
        train_cnt = dist["splits"]["train"]["class_counts"][c]
        val_cnt = dist["splits"]["val"]["class_counts"][c]
        test_cnt = dist["splits"]["test"]["class_counts"][c]
        assert train_cnt > 0, f"Class {c} missing from Train split!"
        assert val_cnt > 0, f"Class {c} missing from Val split!"
        assert test_cnt > 0, f"Class {c} missing from Test split!"
