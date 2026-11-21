# ruff: noqa: E402
"""Phase 7: Final Held-Out Test Evaluation of CustomWaferCNN Baseline.

Executes a single, frozen, leak-free evaluation on the untouched WM-811K test partition (N = 24,705).
- Model: CustomWaferCNN (520,681 parameters)
- Checkpoint: models/checkpoints/exp1_custom_cnn_baseline_best.pt (Epoch 24, Val Macro F1 = 0.8861)
- Input: Single-channel [1, 64, 64], Method C letterbox padding
- Generates:
  - data/interim/figures/final_test_confusion_matrix.png
  - data/interim/figures/final_test_classification_report.json
  - data/interim/figures/final_test_metrics.json
  - data/interim/figures/gradcam/test/ (Representative test Grad-CAM explanations)
"""

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Fix Matplotlib cache warning on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WM811kDataset,
    load_labeled_dataset,
)
from src.data.group_split import load_split_indices
from src.explainability.grad_cam import GradCAM, overlay_heatmap_on_wafer
from src.models.custom_cnn import CustomWaferCNN


def compute_sha256(file_path: Path) -> str:
    """Calculate SHA-256 cryptographic hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    num_bins: int = 15,
) -> dict:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (
            (y_conf > bin_lower) & (y_conf <= bin_upper)
            if i > 0
            else (y_conf >= bin_lower) & (y_conf <= bin_upper)
        )
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidence_in_bin = np.mean(y_conf[in_bin])
            abs_diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

            bin_details.append(
                {
                    "bin_index": i,
                    "range": [round(float(bin_lower), 4), round(float(bin_upper), 4)],
                    "count": int(np.sum(in_bin)),
                    "proportion": round(float(prop_in_bin), 4),
                    "accuracy": round(float(accuracy_in_bin), 4),
                    "confidence": round(float(avg_confidence_in_bin), 4),
                    "error": round(float(abs_diff), 4),
                }
            )

    return {
        "num_bins": num_bins,
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bins": bin_details,
    }


def plot_test_confusion_matrix(
    cm_counts: np.ndarray,
    cm_norm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot dual-panel raw count and normalized confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    # Panel 1: Absolute Counts
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Held-Out Test Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Class", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Panel 2: Normalized Percentages
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title(
        "Held-Out Test Confusion Matrix (Row-Normalized %)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Class", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle(
        "CustomWaferCNN Final Benchmark Evaluation on Held-Out Test Partition (N = 24,705)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    logger.info("=================================================================")
    logger.info("   PHASE 7: FINAL HELD-OUT TEST EVALUATION (CUSTOM WAFER CNN)    ")
    logger.info("=================================================================")

    start_timestamp = datetime.now(UTC).isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # -------------------------------------------------------------------------
    # 1. Verify and Freeze the Selected Model Checkpoint
    # -------------------------------------------------------------------------
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "exp1_custom_cnn_baseline_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Selected model checkpoint not found at {ckpt_path}")

    ckpt_hash = compute_sha256(ckpt_path)
    logger.info(f"Checkpoint Path: {ckpt_path}")
    logger.info(f"Checkpoint SHA-256: {ckpt_hash}")

    t_load_start = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CustomWaferCNN(in_channels=1, num_classes=9).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    t_load_end = time.perf_counter()
    model_loading_time_seconds = t_load_end - t_load_start

    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 0, "Model weights must be completely frozen!"
    logger.info(
        f"Model successfully frozen! Total parameters: {total_params:,} (Trainable: {trainable_params})"
    )
    logger.info(
        f"Checkpoint Best Epoch: {ckpt.get('epoch')} | Recorded Val Macro F1: {ckpt.get('best_score')}"
    )

    # -------------------------------------------------------------------------
    # 2. Test Partition Verification & Integrity
    # -------------------------------------------------------------------------
    data_path = PROJECT_ROOT / "data" / "interim" / "labeled_wafers.pkl"
    splits_path = PROJECT_ROOT / "data" / "interim" / "split_indices.json"

    df = load_labeled_dataset(str(data_path))
    splits = load_split_indices(str(splits_path), "group_stratified_70_15_15")

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Verify counts
    assert len(test_idx) == 24705, f"Expected 24,705 test samples, got {len(test_idx)}"
    test_lots = set(df.iloc[test_idx]["lotName"])
    train_lots = set(df.iloc[train_idx]["lotName"])
    val_lots = set(df.iloc[val_idx]["lotName"])

    assert len(test_lots) == 1537, f"Expected 1,537 test lots, got {len(test_lots)}"
    assert len(train_lots.intersection(test_lots)) == 0, "FATAL: Train-Test lot leakage detected!"
    assert len(val_lots.intersection(test_lots)) == 0, "FATAL: Val-Test lot leakage detected!"
    logger.info("Test Partition Integrity Verified:")
    logger.info(f"  - Total Test Wafers: {len(test_idx):,}")
    logger.info(f"  - Total Test Lots:   {len(test_lots):,}")
    logger.info("  - Zero Lot Overlap with Train or Validation partitions confirmed.")

    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_dataset = WM811kDataset(
        test_df,
        preprocessing_method="method_c",
        representation="single_channel",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------------------------------------------------------
    # 3. Warm-Up & Steady-State Inference Benchmarking
    # -------------------------------------------------------------------------
    logger.info("Running CUDA warm-up (10 forward passes)...")
    dummy_input = torch.zeros(128, 1, 64, 64, device=device)
    t_warm_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t_warm_end = time.perf_counter()
    warmup_time_seconds = t_warm_end - t_warm_start
    logger.info(f"Warm-up complete in {warmup_time_seconds:.4f} seconds.")

    # Reset peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logger.info("Executing held-out test evaluation on complete test set (N = 24,705)...")
    batch_latencies = []
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    t_infer_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            bx = batch["image"].to(device)
            by = batch["label"].numpy()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_b0 = time.perf_counter()

            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_b1 = time.perf_counter()

            batch_latencies.append(t_b1 - t_b0)
            y_true_list.extend(by)
            y_pred_list.extend(preds.cpu().numpy())
            y_probs_list.append(probs.cpu().numpy())

    t_infer_end = time.perf_counter()
    total_inference_time_seconds = t_infer_end - t_infer_start

    # Memory usage
    peak_gpu_allocated_mb = (
        torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    peak_gpu_reserved_mb = (
        torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_probs = np.vstack(y_probs_list)
    y_conf = np.max(y_probs, axis=1)

    # Latency and throughput metrics
    num_samples = len(y_true)
    per_wafer_latencies_ms = np.array(batch_latencies) * 1000 / 128
    avg_batch_latency_ms = float(np.mean(batch_latencies) * 1000)
    avg_wafer_latency_ms = float(np.mean(per_wafer_latencies_ms))
    median_wafer_latency_ms = float(np.median(per_wafer_latencies_ms))
    throughput_wafers_per_sec = float(num_samples / total_inference_time_seconds)

    logger.info("Inference Benchmarks (GPU Steady-State):")
    logger.info(f"  - Total Samples:            {num_samples:,}")
    logger.info(f"  - Total Steady-State Time:  {total_inference_time_seconds:.3f} s")
    logger.info(f"  - Average Batch Latency:    {avg_batch_latency_ms:.3f} ms")
    logger.info(
        f"  - Average Wafer Latency:    {avg_wafer_latency_ms:.4f} ms ({avg_wafer_latency_ms * 1000:.1f} µs)"
    )
    logger.info(f"  - Median Wafer Latency:     {median_wafer_latency_ms:.4f} ms")
    logger.info(f"  - Throughput:               {throughput_wafers_per_sec:.1f} wafers/sec")
    logger.info(f"  - Peak GPU Allocated:       {peak_gpu_allocated_mb:.2f} MB")
    logger.info(f"  - Peak GPU Reserved:        {peak_gpu_reserved_mb:.2f} MB")

    # CPU Benchmark comparison (500 samples)
    logger.info("Benchmarking CPU inference latency on 500 test wafers...")
    cpu_model = CustomWaferCNN(in_channels=1, num_classes=9).to("cpu")
    cpu_model.load_state_dict(ckpt["model_state_dict"])
    cpu_model.eval()

    cpu_sample_tensors = torch.stack([test_dataset[i]["image"] for i in range(500)])
    cpu_loader = torch.utils.data.DataLoader(cpu_sample_tensors, batch_size=32, shuffle=False)
    cpu_latencies = []
    t_cpu_start = time.perf_counter()
    with torch.no_grad():
        for cbx in cpu_loader:
            t0 = time.perf_counter()
            _ = cpu_model(cbx)
            t1 = time.perf_counter()
            cpu_latencies.append((t1 - t0) * 1000 / len(cbx))
    t_cpu_end = time.perf_counter()
    cpu_avg_wafer_latency_ms = float(np.mean(cpu_latencies))
    cpu_throughput = float(500 / (t_cpu_end - t_cpu_start))
    logger.info(f"  - CPU Average Wafer Latency: {cpu_avg_wafer_latency_ms:.4f} ms")
    logger.info(f"  - CPU Throughput:            {cpu_throughput:.1f} wafers/sec")

    # -------------------------------------------------------------------------
    # 4. Global Metrics and Confusion Analysis
    # -------------------------------------------------------------------------
    class_names = [IDX_TO_LABEL[i] for i in range(9)]
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True))

    rep_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(9)),
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0))

    accuracy = float(rep_dict["accuracy"])
    macro_precision = float(rep_dict["macro avg"]["precision"])
    macro_recall = float(rep_dict["macro avg"]["recall"])
    macro_f1 = float(rep_dict["macro avg"]["f1-score"])

    # Clean wafer false alarm analysis
    clean_support = int(cm_counts[0].sum())
    clean_tp = int(cm_counts[0, 0])
    clean_false_alarms = int(clean_support - clean_tp)
    clean_false_alarm_rate = float(clean_false_alarms / clean_support)

    logger.info("=================================================================")
    logger.info("                 FINAL HELD-OUT TEST RESULTS                     ")
    logger.info("=================================================================")
    logger.info(f"Overall Accuracy:       {accuracy * 100:.2f}% ({accuracy:.4f})")
    logger.info(f"Macro F1-Score:         {macro_f1:.4f}")
    logger.info(f"Macro Precision:        {macro_precision:.4f} ({macro_precision * 100:.2f}%)")
    logger.info(f"Macro Recall:           {macro_recall:.4f} ({macro_recall * 100:.2f}%)")
    logger.info(f"Weighted F1-Score:      {weighted_f1:.4f}")
    logger.info(
        f"Clean Wafer False Alarms: {clean_false_alarms} / {clean_support} ({clean_false_alarm_rate * 100:.2f}%)"
    )

    # Per-Class Analysis and Escape Breakdown
    per_class_results = {}
    defect_escape_summary = {}

    for i in range(9):
        cname = IDX_TO_LABEL[i]
        tp = int(cm_counts[i, i])
        fn = int(cm_counts[i].sum() - tp)
        fp = int(cm_counts[:, i].sum() - tp)
        support = int(rep_dict[cname]["support"])
        prec = float(rep_dict[cname]["precision"])
        rec = float(rep_dict[cname]["recall"])
        f1 = float(rep_dict[cname]["f1-score"])

        # Distinguish clean escapes vs other defect misclassifications
        escapes_to_clean = int(cm_counts[i, 0]) if i != 0 else 0
        escapes_to_other_defects = fn - escapes_to_clean if i != 0 else 0
        clean_escape_rate = float(escapes_to_clean / support) if (support > 0 and i != 0) else 0.0

        per_class_results[cname] = {
            "class_index": i,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "escapes_to_clean": escapes_to_clean,
            "escapes_to_other_defects": escapes_to_other_defects,
            "clean_escape_rate": round(clean_escape_rate, 4),
        }

        if i != 0:
            defect_escape_summary[cname] = {
                "support": support,
                "detected_tp": tp,
                "clean_escapes": escapes_to_clean,
                "clean_escape_rate": f"{clean_escape_rate * 100:.2f}%",
                "other_defect_misclassifications": escapes_to_other_defects,
            }

    # Print summary table
    logger.info("\nPer-Class Breakdown on Held-Out Test Set:")
    header = f"{'Class':<10} | {'Support':<7} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Clean Escapes':<16} | {'Other Defect FN':<15}"
    logger.info(header)
    logger.info("-" * len(header))
    for cname, metrics in per_class_results.items():
        esc_str = (
            f"{metrics['escapes_to_clean']} ({metrics['clean_escape_rate'] * 100:.2f}%)"
            if metrics["class_index"] != 0
            else "N/A (clean)"
        )
        other_fn_str = (
            str(metrics["escapes_to_other_defects"])
            if metrics["class_index"] != 0
            else str(metrics["false_negatives"])
        )
        row = f"{cname:<10} | {metrics['support']:<7} | {metrics['true_positives']:<5} | {metrics['false_positives']:<5} | {metrics['false_negatives']:<5} | {metrics['precision']:<7.4f} | {metrics['recall']:<7.4f} | {metrics['f1']:<7.4f} | {esc_str:<16} | {other_fn_str:<15}"
        logger.info(row)

    # -------------------------------------------------------------------------
    # 5. Confidence Distribution and Calibration Analysis
    # -------------------------------------------------------------------------
    correct_mask = y_true == y_pred
    mean_conf_all = float(np.mean(y_conf))
    mean_conf_correct = float(np.mean(y_conf[correct_mask]))
    mean_conf_incorrect = float(np.mean(y_conf[~correct_mask]))

    conf_percentiles = {
        "p10": float(np.percentile(y_conf, 10)),
        "p25": float(np.percentile(y_conf, 25)),
        "median": float(np.percentile(y_conf, 50)),
        "p75": float(np.percentile(y_conf, 75)),
        "p90": float(np.percentile(y_conf, 90)),
        "p99": float(np.percentile(y_conf, 99)),
    }

    calibration = compute_calibration_metrics(y_true, y_pred, y_conf, num_bins=15)
    logger.info("Confidence & Calibration Analysis:")
    logger.info(f"  - Mean Confidence (Overall):   {mean_conf_all * 100:.2f}%")
    logger.info(f"  - Mean Confidence (Correct):   {mean_conf_correct * 100:.2f}%")
    logger.info(f"  - Mean Confidence (Incorrect): {mean_conf_incorrect * 100:.2f}%")
    logger.info(
        f"  - Expected Calibration Error:  {calibration['ece'] * 100:.2f}% (ECE = {calibration['ece']})"
    )
    logger.info(
        f"  - Maximum Calibration Error:   {calibration['mce'] * 100:.2f}% (MCE = {calibration['mce']})"
    )

    # -------------------------------------------------------------------------
    # 6. Save Artifacts
    # -------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "data" / "interim" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix plot
    cm_plot_path = out_dir / "final_test_confusion_matrix.png"
    plot_test_confusion_matrix(cm_counts, cm_norm, class_names, cm_plot_path)
    logger.info(f"Saved confusion matrix plot to: {cm_plot_path}")

    # Full Classification Report JSON
    report_json_path = out_dir / "final_test_classification_report.json"
    with open(report_json_path, "w") as f:
        json.dump(rep_dict, f, indent=2)
    logger.info(f"Saved classification report JSON to: {report_json_path}")

    # Complete Test Metrics JSON
    final_metrics_payload = {
        "evaluation_name": "final_held_out_test_evaluation",
        "timestamp_utc": start_timestamp,
        "model_architecture": "CustomWaferCNN",
        "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_best_epoch": int(ckpt.get("epoch")),
        "checkpoint_val_macro_f1": float(ckpt.get("best_score")),
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "dataset_metadata": {
            "dataset_file": str(data_path.relative_to(PROJECT_ROOT)),
            "splits_file": str(splits_path.relative_to(PROJECT_ROOT)),
            "split_name": "group_stratified_70_15_15",
            "test_sample_count": num_samples,
            "test_lot_count": len(test_lots),
            "lot_leakage_verified": True,
        },
        "system_environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "None",
            "os_platform": platform.platform(),
        },
        "global_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "clean_wafer_false_alarms": clean_false_alarms,
            "clean_wafer_total": clean_support,
            "clean_wafer_false_alarm_rate": round(clean_false_alarm_rate, 4),
        },
        "efficiency_benchmarks": {
            "model_loading_seconds": round(model_loading_time_seconds, 4),
            "warmup_seconds": round(warmup_time_seconds, 4),
            "total_gpu_inference_seconds": round(total_inference_time_seconds, 4),
            "gpu_average_batch_latency_ms": round(avg_batch_latency_ms, 3),
            "gpu_average_wafer_latency_ms": round(avg_wafer_latency_ms, 4),
            "gpu_median_wafer_latency_ms": round(median_wafer_latency_ms, 4),
            "gpu_throughput_wafers_per_sec": round(throughput_wafers_per_sec, 2),
            "gpu_peak_allocated_vram_mb": round(peak_gpu_allocated_mb, 2),
            "gpu_peak_reserved_vram_mb": round(peak_gpu_reserved_mb, 2),
            "cpu_average_wafer_latency_ms": round(cpu_avg_wafer_latency_ms, 4),
            "cpu_throughput_wafers_per_sec": round(cpu_throughput, 2),
        },
        "confidence_and_calibration": {
            "mean_confidence_overall": round(mean_conf_all, 4),
            "mean_confidence_correct": round(mean_conf_correct, 4),
            "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
            "percentiles": conf_percentiles,
            "calibration": calibration,
        },
        "per_class_metrics": per_class_results,
        "defect_escapes_summary": defect_escape_summary,
        "confusion_matrix": cm_counts.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
    }

    metrics_json_path = out_dir / "final_test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(final_metrics_payload, f, indent=2)
    logger.info(f"Saved complete test metrics JSON to: {metrics_json_path}")

    # -------------------------------------------------------------------------
    # 7. Representative Test-Set Grad-CAM Visualizations
    # -------------------------------------------------------------------------
    logger.info("Generating representative test-set Grad-CAM visualizations...")
    test_gradcam_dir = PROJECT_ROOT / "data" / "interim" / "figures" / "gradcam" / "test"
    test_gradcam_dir.mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model=model, target_layer="features.conv_block4", device=device)

    # Find highest confidence correct sample for each class in test set
    test_correct_samples = {}
    test_scratch_escape = None
    test_loc_escape = None

    for idx in range(num_samples):
        gt = int(y_true[idx])
        pred = int(y_pred[idx])
        conf = float(y_conf[idx])

        if gt == pred:
            if gt not in test_correct_samples or conf > test_correct_samples[gt][1]:
                test_correct_samples[gt] = (idx, conf, pred)
        elif (
            gt == LABEL_TO_IDX["Scratch"]
            and pred == LABEL_TO_IDX["none"]
            and test_scratch_escape is None
        ):
            test_scratch_escape = (idx, conf, pred)
        elif gt == LABEL_TO_IDX["Loc"] and pred == LABEL_TO_IDX["none"] and test_loc_escape is None:
            test_loc_escape = (idx, conf, pred)

    # 9-class composite summary grid for test set
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(9):
        cls_name = IDX_TO_LABEL[i]
        if i in test_correct_samples:
            t_idx, conf, pred = test_correct_samples[i]
            sample = test_dataset[t_idx]
            wt = sample["image"]
            wnp = wt.squeeze().cpu().numpy().astype(np.uint8)

            cam_map, _, _ = cam_engine.generate_cam(wt.unsqueeze(0), target_class=pred)
            overlay = overlay_heatmap_on_wafer(
                wnp, cam_map.squeeze().numpy(), alpha=0.5, colormap="jet"
            )

            axes[i].imshow(overlay)
            axes[i].set_title(
                f"Test {i}: {cls_name}\nConf: {conf * 100:.1f}% [CORRECT]",
                fontsize=11,
                fontweight="bold",
            )
            axes[i].axis("off")

    plt.suptitle(
        "CustomWaferCNN Grad-CAM Visualizations on Held-Out Test Set (All 9 Classes)",
        fontsize=15,
        fontweight="bold",
        y=0.94,
    )
    plt.tight_layout()
    test_grid_path = test_gradcam_dir / "test_set_summary_9classes.png"
    fig.savefig(test_grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved test Grad-CAM composite grid to: {test_grid_path}")

    # Single test escape figure if found
    if test_scratch_escape is not None:
        s_idx, conf, pred = test_scratch_escape
        sample = test_dataset[s_idx]
        wt = sample["image"]
        wnp = wt.squeeze().cpu().numpy().astype(np.uint8)

        cam_map, _, _ = cam_engine.generate_cam(wt.unsqueeze(0), target_class=pred)
        overlay = overlay_heatmap_on_wafer(
            wnp, cam_map.squeeze().numpy(), alpha=0.5, colormap="jet"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title(
            f"Test Set Escape: True Scratch predicted as none ({conf * 100:.1f}%)\n[HELD-OUT TEST SAMPLE]",
            fontsize=11,
            fontweight="bold",
            color="red",
        )
        ax.axis("off")
        esc_fig_path = test_gradcam_dir / "test_set_escape_scratch_pred_none.png"
        fig.savefig(esc_fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved test defect escape figure to: {esc_fig_path}")

    cam_engine.remove_hooks()
    logger.info("Phase 7 Final Held-Out Test Evaluation successfully completed!")


if __name__ == "__main__":
    main()
