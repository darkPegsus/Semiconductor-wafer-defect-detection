"""Data engineering, dataset abstractions, group-aware splitting, and preprocessing."""

from src.data.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WM811kDataset,
    extract_labeled_dataset,
    load_labeled_dataset,
)
from src.data.group_split import (
    compute_split_distributions,
    create_stratified_group_split,
    extract_original_benchmark_split,
    load_split_indices,
    save_split_indices,
    verify_zero_lot_leakage,
)
from src.data.preprocessor import (
    WaferPreprocessor,
    compute_transformation_metrics,
    generate_ablation_visualizations,
    preprocess_direct_resize,
    preprocess_fixed_canvas_padding,
    preprocess_letterbox_padding,
)
from src.data.transforms import (
    WaferTransform,
    encode_one_hot,
    encode_single_channel,
)
from src.data.validator import (
    extract_scalar_label,
    generate_validation_report,
    load_raw_lswmd,
    validate_and_extract_metadata,
)

__all__ = [
    "IDX_TO_LABEL",
    "LABEL_TO_IDX",
    "WM811kDataset",
    "WaferPreprocessor",
    "WaferTransform",
    "compute_split_distributions",
    "compute_transformation_metrics",
    "create_stratified_group_split",
    "encode_one_hot",
    "encode_single_channel",
    "extract_labeled_dataset",
    "extract_original_benchmark_split",
    "extract_scalar_label",
    "generate_ablation_visualizations",
    "generate_validation_report",
    "load_labeled_dataset",
    "load_raw_lswmd",
    "load_split_indices",
    "preprocess_direct_resize",
    "preprocess_fixed_canvas_padding",
    "preprocess_letterbox_padding",
    "save_split_indices",
    "validate_and_extract_metadata",
    "verify_zero_lot_leakage",
]
