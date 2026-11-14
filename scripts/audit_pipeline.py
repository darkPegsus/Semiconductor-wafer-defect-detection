"""Data pipeline audit script to verify split integrity, lot isolation, and class representation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset import load_labeled_dataset
from src.data.group_split import (
    compute_split_distributions,
    load_split_indices,
    verify_zero_lot_leakage,
)


def run_audit():
    print("Loading labeled interim dataset...")
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")
    orig_splits = load_split_indices("data/interim/split_indices.json", "original_benchmark")

    tr_idx = splits["train"]
    va_idx = splits["val"]
    te_idx = splits["test"]

    print("=== 1. Index Mutual Exclusivity ===")
    set_tr, set_va, set_te = set(tr_idx), set(va_idx), set(te_idx)
    overlap_tr_va = set_tr & set_va
    overlap_tr_te = set_tr & set_te
    overlap_va_te = set_va & set_te
    total_count = len(tr_idx) + len(va_idx) + len(te_idx)
    print(f"Train indices: {len(tr_idx):>7d}")
    print(f"Val indices:   {len(va_idx):>7d}")
    print(f"Test indices:  {len(te_idx):>7d}")
    print(
        f"Total indexed: {total_count:>7d} | Matches labeled df (172950): {total_count == len(df)}"
    )
    print(
        f"Overlaps -> Train-Val: {len(overlap_tr_va)}, Train-Test: {len(overlap_tr_te)}, Val-Test: {len(overlap_va_te)}"
    )

    print("\n=== 2. Manufacturing Lots Overlap Verification ===")
    zero_leak = verify_zero_lot_leakage(df, tr_idx, va_idx, te_idx)
    print(f"Zero lot leakage verification: {zero_leak}")
    lots_tr = set(df.iloc[tr_idx]["lotName"])
    lots_va = set(df.iloc[va_idx]["lotName"])
    lots_te = set(df.iloc[te_idx]["lotName"])
    total_unique_lots = len(lots_tr | lots_va | lots_te)
    print(f"Unique Lots -> Train: {len(lots_tr)}, Val: {len(lots_va)}, Test: {len(lots_te)}")
    print(
        f"Union of lots: {total_unique_lots} | Matches dataset lots ({df['lotName'].nunique()}): {total_unique_lots == df['lotName'].nunique()}"
    )

    print("\n=== 3. Stratification & Class Representation ===")
    dist = compute_split_distributions(df, splits)
    summary = []
    classes = sorted(df["failureType"].unique())
    for c in classes:
        tot = dist["per_class_summary"][c]["total"]
        tr_cnt = dist["splits"]["train"]["class_counts"][c]
        va_cnt = dist["splits"]["val"]["class_counts"][c]
        te_cnt = dist["splits"]["test"]["class_counts"][c]
        summary.append(
            {
                "Class": c,
                "Total": tot,
                "Train": tr_cnt,
                "Train_%": round(tr_cnt / tot * 100, 2),
                "Val": va_cnt,
                "Val_%": round(va_cnt / tot * 100, 2),
                "Test": te_cnt,
                "Test_%": round(te_cnt / tot * 100, 2),
                "Min Split Count": min(tr_cnt, va_cnt, te_cnt),
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n=== 4. Original Author Benchmark Preservation ===")
    orig_tr_idx = orig_splits["train"]
    orig_te_idx = orig_splits["test"]
    orig_total = len(orig_tr_idx) + len(orig_te_idx)
    orig_tr_lots = set(df.iloc[orig_tr_idx]["lotName"])
    orig_te_lots = set(df.iloc[orig_te_idx]["lotName"])
    overlap_orig = orig_tr_lots & orig_te_lots
    print(f"Original Benchmark -> Train: {len(orig_tr_idx)}, Test: {len(orig_te_idx)}")
    print(
        f"Sum of original benchmark indices: {orig_total} (Matches 172950: {orig_total == len(df)})"
    )
    print(
        f"Original Lots -> Train: {len(orig_tr_lots)}, Test: {len(orig_te_lots)}, Overlap: {len(overlap_orig)}"
    )

    print("\n=== 5. Labels and Die Value Consistency ===")
    print(f"Unique failure classes ({len(classes)}): {classes}")
    sample_indices = np.random.RandomState(42).choice(len(df), size=2000, replace=False)
    unique_dies = set()
    for idx in sample_indices:
        wm = df["waferMap"].iloc[idx]
        unique_dies.update(np.unique(wm).tolist())
    print(
        f"Unique die values across 2000 random wafers: {sorted(unique_dies)} (Expected [0, 1, 2])"
    )


if __name__ == "__main__":
    run_audit()
