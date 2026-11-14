"""Comprehensive multi-class evaluation metrics for imbalanced wafer defect classification."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.dataset import IDX_TO_LABEL


def compute_metrics(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
    class_names: dict[int, str] | list[str] | None = None,
) -> dict[str, Any]:
    """Calculate accuracy, macro scores, per-class F1/precision/recall, and confusion matrix.

    Args:
        y_true: Ground-truth integer labels [N].
        y_pred: Predicted integer class labels [N] or logits [N, C].
        class_names: Mapping or list of class names (defaults to IDX_TO_LABEL).

    Returns:
        Structured dictionary of evaluation metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true_arr = y_true.detach().cpu().numpy()
    else:
        y_true_arr = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        if y_pred.ndim == 2:
            y_pred_arr = torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
        else:
            y_pred_arr = y_pred.detach().cpu().numpy()
    else:
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 2:
            y_pred_arr = np.argmax(y_pred_arr, axis=-1)

    if class_names is None:
        idx_mapping = IDX_TO_LABEL
        classes = sorted(idx_mapping.keys())
    elif isinstance(class_names, dict):
        idx_mapping = class_names
        classes = sorted(idx_mapping.keys())
    else:
        idx_mapping = dict(enumerate(class_names))
        classes = list(range(len(class_names)))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))

    # Per-class metrics
    per_cls_prec = precision_score(
        y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0.0
    )
    per_cls_rec = recall_score(
        y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0.0
    )
    per_cls_f1 = f1_score(y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0.0)

    per_class_summary: dict[str, dict[str, float]] = {}
    for i, c in enumerate(classes):
        name = idx_mapping.get(c, str(c))
        per_class_summary[name] = {
            "precision": round(float(per_cls_prec[i]), 4),
            "recall": round(float(per_cls_rec[i]), 4),
            "f1": round(float(per_cls_f1[i]), 4),
            "support": int(np.sum(y_true_arr == c)),
        }

    # Confusion matrix
    cm_counts = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_summary,
        "confusion_matrix": cm_counts.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }
