"""Group-aware data splitting with multi-class stratification and zero lot leakage.

Implements:
1. Stratified Group Partitioning (70% Train, 15% Val, 15% Test) with lot isolation
   using StratifiedGroupKFold on manufacturing lots.
2. Extraction and preservation of the original author benchmark split.
3. Verification tools for zero lot leakage and class distribution parity.
4. JSON serialization of reproducible split indices.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold


def create_stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "lotName",
    target_col: str = "failureType",
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Partition dataset into ~70% Train, 15% Val, 15% Test with zero lot leakage.

    Uses 7-fold StratifiedGroupKFold where 5 folds form Train (~71.4%), 1 fold
    forms Val (~14.3%), and 1 fold forms Test (~14.3%).

    Args:
        df: DataFrame containing labeled wafer records.
        group_col: Column name identifying manufacturing lot groups.
        target_col: Column name identifying categorical failure classes.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to 1D integer index arrays.
    """
    logger.info(
        f"Creating stratified group split on {len(df)} records "
        f"(groups={group_col}, target={target_col}, seed={random_state})..."
    )

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df, df[target_col], df[group_col]))

    # Fold 5 as Val, Fold 6 as Test, Folds 0..4 as Train
    val_indices = splits[5][1]
    test_indices = splits[6][1]

    # Everything outside val and test belongs to train
    n_samples = len(df)
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[val_indices] = False
    train_mask[test_indices] = False
    train_indices = np.where(train_mask)[0]

    # Verify zero lot leakage
    verify_zero_lot_leakage(df, train_indices, val_indices, test_indices, group_col)

    logger.info(
        f"Split complete: Train={len(train_indices)} ({len(train_indices) / n_samples * 100:.2f}%), "
        f"Val={len(val_indices)} ({len(val_indices) / n_samples * 100:.2f}%), "
        f"Test={len(test_indices)} ({len(test_indices) / n_samples * 100:.2f}%)"
    )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def extract_original_benchmark_split(
    df: pd.DataFrame,
    split_col: str = "originalSplit",
    group_col: str = "lotName",
) -> dict[str, np.ndarray]:
    """Extract author-provided original benchmark split indices.

    Args:
        df: DataFrame containing labeled wafer records.
        split_col: Column containing original split labels ('Training' / 'Test').
        group_col: Column name identifying manufacturing lot groups.

    Returns:
        Dictionary mapping 'train' and 'test' to 1D integer index arrays.
    """
    if split_col not in df.columns:
        raise KeyError(f"Column '{split_col}' not found in dataframe.")

    train_indices = np.where(df[split_col] == "Training")[0]
    test_indices = np.where(df[split_col] == "Test")[0]

    # Verify zero lot leakage in original split
    train_lots = set(df.iloc[train_indices][group_col])
    test_lots = set(df.iloc[test_indices][group_col])
    overlap = train_lots.intersection(test_lots)

    logger.info(
        f"Original benchmark split: Train={len(train_indices)} (lots={len(train_lots)}), "
        f"Test={len(test_indices)} (lots={len(test_lots)}), Overlap={len(overlap)}"
    )

    return {
        "train": train_indices,
        "test": test_indices,
    }


def verify_zero_lot_leakage(
    df: pd.DataFrame,
    train_idx: Sequence[int] | np.ndarray,
    val_idx: Sequence[int] | np.ndarray,
    test_idx: Sequence[int] | np.ndarray,
    group_col: str = "lotName",
) -> bool:
    """Verify strictly zero group (lot) overlap across train, val, and test subsets.

    Raises:
        AssertionError: If any lot appears in more than one partition.

    Returns:
        True if all subsets are completely disjoint.
    """
    train_lots = set(df.iloc[train_idx][group_col])
    val_lots = set(df.iloc[val_idx][group_col])
    test_lots = set(df.iloc[test_idx][group_col])

    overlap_tr_va = train_lots.intersection(val_lots)
    overlap_tr_te = train_lots.intersection(test_lots)
    overlap_va_te = val_lots.intersection(test_lots)

    if overlap_tr_va or overlap_tr_te or overlap_va_te:
        msg = (
            f"Lot leakage detected! "
            f"Train-Val overlap: {len(overlap_tr_va)}, "
            f"Train-Test overlap: {len(overlap_tr_te)}, "
            f"Val-Test overlap: {len(overlap_va_te)}"
        )
        logger.error(msg)
        raise AssertionError(msg)

    logger.info("Zero lot leakage verified: Train, Val, and Test lots are 100% disjoint.")
    return True


def compute_split_distributions(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    target_col: str = "failureType",
    group_col: str = "lotName",
) -> dict[str, Any]:
    """Compute exact per-class and per-lot distributions for all splits.

    Args:
        df: DataFrame containing dataset.
        splits: Mapping of split name ('train', 'val', 'test') to index arrays.
        target_col: Target class column.
        group_col: Lot identifier column.

    Returns:
        Structured dictionary with class counts, proportions, and lot metrics.
    """
    classes = sorted(df[target_col].unique())
    total_samples = len(df)
    results: dict[str, Any] = {
        "total_samples": total_samples,
        "splits": {},
        "per_class_summary": {},
    }

    for split_name, idx in splits.items():
        subset = df.iloc[idx]
        split_lots = set(subset[group_col])
        counts = subset[target_col].value_counts().to_dict()
        results["splits"][split_name] = {
            "sample_count": len(idx),
            "sample_percentage": round(len(idx) / total_samples * 100, 2),
            "unique_lots": len(split_lots),
            "class_counts": {c: int(counts.get(c, 0)) for c in classes},
        }

    for c in classes:
        c_tot = int((df[target_col] == c).sum())
        entry: dict[str, Any] = {"total": c_tot}
        for split_name, _idx in splits.items():
            cnt = results["splits"][split_name]["class_counts"][c]
            pct = round(cnt / c_tot * 100, 2) if c_tot > 0 else 0.0
            entry[f"{split_name}_count"] = cnt
            entry[f"{split_name}_pct"] = pct
        results["per_class_summary"][c] = entry

    return results


def save_split_indices(
    group_stratified_splits: dict[str, np.ndarray],
    original_benchmark_splits: dict[str, np.ndarray],
    output_path: str | Path = "data/interim/split_indices.json",
    random_state: int = 42,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save reproducible split indices to a JSON file.

    Args:
        group_stratified_splits: Stratified group split dictionary.
        original_benchmark_splits: Original author split dictionary.
        output_path: Path to output JSON file.
        random_state: Random state seed.
        metadata: Optional metadata dictionary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "random_state": random_state,
        "metadata": metadata or {},
        "group_stratified_70_15_15": {
            k: v.tolist() if isinstance(v, np.ndarray) else list(v)
            for k, v in group_stratified_splits.items()
        },
        "original_benchmark": {
            k: v.tolist() if isinstance(v, np.ndarray) else list(v)
            for k, v in original_benchmark_splits.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Split indices saved to: {output_path}")


def load_split_indices(
    path: str | Path = "data/interim/split_indices.json",
    split_type: str = "group_stratified_70_15_15",
) -> dict[str, np.ndarray]:
    """Load split index arrays from JSON file.

    Args:
        path: Path to JSON split indices file.
        split_type: 'group_stratified_70_15_15' or 'original_benchmark'.

    Returns:
        Dictionary of partition names to numpy integer index arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split indices file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if split_type not in data:
        raise KeyError(f"Split type '{split_type}' not found. Available: {list(data.keys())}")

    return {k: np.array(v, dtype=np.int64) for k, v in data[split_type].items()}


if __name__ == "__main__":
    from src.data.dataset import load_labeled_dataset

    df = load_labeled_dataset()
    strat_splits = create_stratified_group_split(df, random_state=42)
    orig_splits = extract_original_benchmark_split(df)

    dist_strat = compute_split_distributions(df, strat_splits)
    dist_orig = compute_split_distributions(df, orig_splits)

    save_split_indices(
        group_stratified_splits=strat_splits,
        original_benchmark_splits=orig_splits,
        random_state=42,
        metadata={
            "group_stratified_summary": dist_strat["per_class_summary"],
            "original_benchmark_summary": dist_orig["per_class_summary"],
        },
    )
