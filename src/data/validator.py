"""Dataset schema validator and empirical statistics extractor for WM-811K (LSWMD).

Handles legacy Python 2 pickle loading, verifies die encodings, schema structure,
computes empirical class and wafer dimension distributions, and outputs a structured
validation report JSON.
"""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.core.indexes
from loguru import logger

# Compatibility patch for legacy Pandas pickles referencing pandas.indexes
if "pandas.indexes" not in sys.modules:
    sys.modules["pandas.indexes"] = pandas.core.indexes


def extract_scalar_label(val: Any) -> str:
    """Extract string label from nested numpy array/list or return 'unlabeled'."""
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return "unlabeled"
        inner = val[0]
        if isinstance(inner, (np.ndarray, list)):
            if len(inner) == 0:
                return "unlabeled"
            return str(inner[0])
        return str(inner)
    if pd.isna(val):
        return "unlabeled"
    return str(val)


def load_raw_lswmd(raw_path: str | Path) -> pd.DataFrame:
    """Load raw LSWMD.pkl with Python 2 compatibility decoding.

    Args:
        raw_path: Path to raw LSWMD.pkl file.

    Returns:
        DataFrame containing all 811,457 records.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    logger.info(f"Loading raw dataset from {raw_path}...")
    with open(raw_path, "rb") as f:
        # encoding='latin1' is required because LSWMD.pkl was pickled with Python 2
        df = pickle.load(f, encoding="latin1")

    logger.info(f"Successfully loaded DataFrame with shape: {df.shape}")
    return df


def validate_and_extract_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect dataset, validate die values, extract class frequencies and dimensions.

    Args:
        df: Raw LSWMD DataFrame.

    Returns:
        Dictionary containing comprehensive validation metadata.
    """
    logger.info("Extracting empirical dataset validation metrics...")

    total_records = len(df)
    columns = list(df.columns)

    # Standardize label strings
    labels = df["failureType"].apply(extract_scalar_label)
    all_class_counts = labels.value_counts(dropna=False).to_dict()

    labeled_mask = labels != "unlabeled"
    labeled_df = df[labeled_mask].copy()
    labeled_df["clean_label"] = labels[labeled_mask]
    num_labeled = len(labeled_df)
    num_unlabeled = total_records - num_labeled

    labeled_class_counts = labeled_df["clean_label"].value_counts().to_dict()
    labeled_class_percentages = {
        k: round(v / num_labeled * 100, 4) for k, v in labeled_class_counts.items()
    }

    # Wafer map dimension analysis on labeled subset
    shapes = [wm.shape for wm in labeled_df["waferMap"]]
    heights = [s[0] for s in shapes]
    widths = [s[1] for s in shapes]
    aspect_ratios = [s[1] / s[0] for s in shapes]

    unique_shapes_count = len(set(shapes))
    shape_counter = Counter(shapes)
    top_10_shapes = [
        {
            "shape": f"{s[0]}x{s[1]}",
            "height": s[0],
            "width": s[1],
            "count": count,
            "percentage": round(count / num_labeled * 100, 2),
        }
        for s, count in shape_counter.most_common(10)
    ]

    # Inspect unique die values across a sample of labeled wafers
    sample_size = min(5000, num_labeled)
    unique_die_values = set()
    for wm in labeled_df["waferMap"].iloc[:sample_size]:
        unique_die_values.update(np.unique(wm).tolist())

    # Lot analysis on labeled subset
    lot_counts = labeled_df["lotName"].value_counts()
    unique_lots = len(lot_counts)

    # Inspect raw author split
    raw_splits = df["trianTestLabel"].apply(extract_scalar_label)
    raw_split_counts = raw_splits.value_counts().to_dict()

    report: dict[str, Any] = {
        "dataset_name": "WM-811K (LSWMD)",
        "total_records": total_records,
        "total_columns": len(columns),
        "column_names": columns,
        "labeled_records": num_labeled,
        "unlabeled_records": num_unlabeled,
        "labeled_percentage": round(num_labeled / total_records * 100, 2),
        "all_records_class_distribution": all_class_counts,
        "die_encodings": {
            "verified_values": sorted(unique_die_values),
            "mapping": {0: "exterior", 1: "normal_die", 2: "defect_die"},
        },
        "labeled_class_distribution": {
            "counts": labeled_class_counts,
            "percentages": labeled_class_percentages,
        },
        "dimension_statistics": {
            "unique_shapes": unique_shapes_count,
            "height": {
                "min": int(min(heights)),
                "max": int(max(heights)),
                "mean": round(float(np.mean(heights)), 2),
                "median": float(np.median(heights)),
                "percentiles": {
                    "p5": float(np.percentile(heights, 5)),
                    "p25": float(np.percentile(heights, 25)),
                    "p50": float(np.percentile(heights, 50)),
                    "p75": float(np.percentile(heights, 75)),
                    "p95": float(np.percentile(heights, 95)),
                },
            },
            "width": {
                "min": int(min(widths)),
                "max": int(max(widths)),
                "mean": round(float(np.mean(widths)), 2),
                "median": float(np.median(widths)),
                "percentiles": {
                    "p5": float(np.percentile(widths, 5)),
                    "p25": float(np.percentile(widths, 25)),
                    "p50": float(np.percentile(widths, 50)),
                    "p75": float(np.percentile(widths, 75)),
                    "p95": float(np.percentile(widths, 95)),
                },
            },
            "aspect_ratio": {
                "min": round(float(min(aspect_ratios)), 2),
                "max": round(float(max(aspect_ratios)), 2),
                "median": round(float(np.median(aspect_ratios)), 2),
            },
            "top_10_most_frequent_shapes": top_10_shapes,
        },
        "lot_statistics": {
            "total_unique_lots": unique_lots,
            "wafers_per_lot": {
                "min": int(lot_counts.min()),
                "max": int(lot_counts.max()),
                "mean": round(float(lot_counts.mean()), 2),
                "median": float(lot_counts.median()),
                "exact_25_wafer_lots": int((lot_counts == 25).sum()),
                "exact_25_percentage": round(
                    float((lot_counts == 25).sum() / unique_lots * 100), 2
                ),
            },
        },
        "raw_author_split": {
            "counts": raw_split_counts,
        },
    }

    logger.info("Validation metrics successfully extracted.")
    return report


def generate_validation_report(
    raw_path: str | Path = "data/raw/LSWMD.pkl",
    output_json: str | Path = "data/interim/dataset_validation_report.json",
) -> dict[str, Any]:
    """Load raw dataset, compute validation metrics, and save report to JSON."""
    raw_path = Path(raw_path)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_lswmd(raw_path)
    report = validate_and_extract_metadata(df)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to: {output_json}")
    return report


if __name__ == "__main__":
    generate_validation_report()
