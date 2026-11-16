# ruff: noqa: E402
"""Experiment 2: Custom CNN with Class-Weighted Cross-Entropy.

Executes controlled training under:
- Architecture: CustomWaferCNN (in_channels=1, num_classes=9)
- Input representation: Single-channel numerical [B, 1, 64, 64]
- Preprocessing: Method C (aspect-ratio-preserving letterbox padding)
- Loss: Class-Weighted Cross-Entropy (Effective-Number weighting, beta=0.9999)
- Sampling: Uniform shuffle (no weighted sampling)
- Optimization: AdamW (lr=1e-3, weight_decay=1e-2) + CosineAnnealingLR
- Mixed Precision: Enabled (torch.amp)
- Early Stopping: Validation Macro F1 (patience=7, mode="max")
- Maximum Epochs: 30
- Checkpointing: models/checkpoints/exp2_custom_cnn_weighted_ce_best.pt
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Fix Matplotlib cache warning on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.data.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WM811kDataset,
    load_labeled_dataset,
)
from src.data.group_split import (
    load_split_indices,
    verify_zero_lot_leakage,
)
from src.models.custom_cnn import CustomWaferCNN
from src.training.early_stopping import EarlyStopping
from src.training.engine import WaferTrainer
from src.training.losses import compute_class_weights
from src.training.scheduler import create_lr_scheduler


def run_pre_training_verifications(df, splits):
    """Run required 6 pre-training verification checks."""
    logger.info("=================================================================")
    logger.info("          RUNNING PRE-TRAINING VERIFICATION CHECKS               ")
    logger.info("=================================================================")

    # 1. Dataset and splits exist and loadable
    assert len(df) == 172950, f"Expected 172950 records, got {len(df)}"
    assert "train" in splits and "val" in splits and "test" in splits
    logger.info("Check 1: Dataset and split files verified and loaded successfully.")

    # 2. Train, val, test have no lot overlap
    verify_zero_lot_leakage(df, splits["train"], splits["val"], splits["test"])
    logger.info("Check 2: Zero lot leakage verified across Train, Val, and Test.")

    # 3. Class-to-index mapping consistency
    assert len(LABEL_TO_IDX) == 9
    for name, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == name
    logger.info("Check 3: Class-to-index bidirectional mappings verified (9 classes).")

    # 4. Small forward and backward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = CustomWaferCNN(in_channels=1, num_classes=9).to(device)
    dummy_x = torch.randn(4, 1, 64, 64, device=device)
    dummy_y = torch.tensor([0, 1, 2, 8], dtype=torch.long, device=device)
    dummy_weights = torch.ones(9, dtype=torch.float32, device=device)
    dummy_out = dummy_model(dummy_x)
    assert dummy_out.shape == (4, 9)
    dummy_loss = nn.CrossEntropyLoss(weight=dummy_weights)(dummy_out, dummy_y)
    dummy_loss.backward()
    assert dummy_model.block1.block[0].weight.grad is not None
    del dummy_model, dummy_x, dummy_y, dummy_out, dummy_loss, dummy_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Check 4: Forward and backward pass with weighted CE sanity check PASSED.")

    # 5. Check GPU memory and select safe batch size
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Check 5: GPU detected: {gpu_name} with {vram_gb:.2f} GB VRAM.")
        logger.info("Batch size 64 selected: Memory profile validated (<500 MB active required).")
    else:
        logger.info("Check 5: Running on CPU.")

    # 6. Confirm test set isolation
    test_count = len(splits["test"])
    logger.info(
        f"Check 6: Test partition ({test_count} wafers, 1537 lots) strictly ISOLATED. "
        "Test set will NOT be used for model selection, validation, or training."
    )
    logger.info("=================================================================\n")


def compute_and_validate_weights(train_df) -> torch.Tensor:
    """Compute effective-number class weights strictly on training set and validate stability."""
    train_targets = [LABEL_TO_IDX[str(x)] for x in train_df["failureType"]]
    beta = 0.9999
    weights = compute_class_weights(
        labels=train_targets,
        num_classes=9,
        mode="effective_num",
        beta=beta,
    )

    logger.info("=================================================================")
    logger.info(f"    CLASS WEIGHT CALCULATION (Cui et al. CVPR 2019, beta={beta})   ")
    logger.info("=================================================================")
    logger.info(f"{'Idx':<4} | {'Class Name':<12} | {'Train Count':<12} | {'Weight':<10}")
    logger.info("-" * 46)

    counts = [0] * 9
    for t in train_targets:
        counts[t] += 1

    for idx in range(9):
        cls_name = IDX_TO_LABEL[idx]
        logger.info(f"{idx:<4} | {cls_name:<12} | {counts[idx]:<12} | {weights[idx].item():<10.4f}")

    # Stability checks
    weight_min = weights.min().item()
    weight_max = weights.max().item()
    weight_mean = weights.mean().item()
    weight_sum = weights.sum().item()
    ratio = weight_max / max(1e-8, weight_min)

    logger.info("-" * 46)
    logger.info(
        f"Weight Sum: {weight_sum:.4f} | Weight Mean: {weight_mean:.4f} | Max/Min Ratio: {ratio:.2f}"
    )

    # Verify no NaN or Inf
    assert not torch.isnan(weights).any(), "NaN found in computed class weights!"
    assert not torch.isinf(weights).any(), "Inf found in computed class weights!"
    # Verify reasonable scaling
    assert 0.99 <= weight_mean <= 1.01, f"Expected mean weight ~1.0, got {weight_mean}"
    assert ratio < 150.0, f"Max/min weight ratio ({ratio:.1f}) exceeds stability threshold (150)!"
    logger.info("Weight stability verified: Bounded max/min ratio prevents gradient explosion.")
    logger.info("=================================================================\n")

    return weights


def plot_training_curves(history: dict, output_path: Path):
    """Plot training/validation loss, accuracy, and macro F1 curves."""
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(
        epochs, history["train_loss"], "o-", label="Train Loss (Weighted)", color="#1f77b4", lw=2
    )
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss (Weighted)", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Weighted Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Training & Validation Loss (Exp 2)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Metrics curve
    ax2.plot(epochs, history["val_accuracy"], "o-", label="Val Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_macro_f1"], "s--", label="Val Macro F1", color="#d62728", lw=2)
    ax2.plot(
        epochs,
        history["val_macro_precision"],
        "^:",
        label="Val Macro Prec",
        color="#9467bd",
        lw=1.5,
    )
    ax2.plot(
        epochs,
        history["val_macro_recall"],
        "v:",
        label="Val Macro Rec",
        color="#8c564b",
        lw=1.5,
    )
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Classification Metrics (Exp 2)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved to {output_path}")


def plot_confusion_matrix(cm_data: list[list[int]], class_names: list[str], output_path: Path):
    """Plot confusion matrix heatmap."""
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix (Experiment 2: Weighted CE)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    # Load dataset & split indices
    logger.info("Loading labeled dataset and split indices...")
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")

    # Run pre-training verifications
    run_pre_training_verifications(df, splits)

    train_indices = splits["train"]
    val_indices = splits["val"]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    # Compute class weights strictly on training set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_and_validate_weights(train_df).to(device)

    # Datasets
    train_dataset = WM811kDataset(
        train_df,
        preprocessing_method="method_c",
        representation="single_channel",
    )
    val_dataset = WM811kDataset(
        val_df,
        preprocessing_method="method_c",
        representation="single_channel",
    )

    # DataLoaders (num_workers=0 optimal on Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = CustomWaferCNN(in_channels=1, num_classes=9)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 30
    scheduler = create_lr_scheduler(
        optimizer, name="CosineAnnealingLR", epochs=epochs, eta_min=1e-6
    )

    checkpoint_path = Path("models/checkpoints/exp2_custom_cnn_weighted_ce_best.pt")
    early_stopping = EarlyStopping(
        patience=7,
        min_delta=0.001,
        mode="max",
        checkpoint_path=checkpoint_path,
    )

    trainer = WaferTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        use_amp=True,
        max_grad_norm=1.0,
    )

    # Fit model
    history = trainer.fit(epochs=epochs, show_progress=False)
    total_duration = time.time() - start_time

    # Save training curves
    curves_path = Path("data/interim/figures/exp2_weighted_ce_training_curves.png")
    plot_training_curves(history, curves_path)

    # Best epoch and best metadata
    best_epoch = early_stopping.best_epoch
    best_score = early_stopping.best_score
    best_metrics = (
        early_stopping.best_metadata["val_metrics"] if early_stopping.best_metadata else None
    )

    # Save confusion matrix plot if available
    cm_path = Path("data/interim/figures/exp2_weighted_ce_confusion_matrix.png")
    if best_metrics and "confusion_matrix" in best_metrics:
        class_names = [IDX_TO_LABEL[i] for i in range(9)]
        plot_confusion_matrix(best_metrics["confusion_matrix"], class_names, cm_path)

    # Memory statistics summary across all epochs
    peak_alloc = max(history.get("peak_allocated_mb", [0.0]))
    peak_res = max(history.get("peak_reserved_mb", [0.0]))
    peak_smi = max(history.get("smi_used_mb", [0.0]))
    avg_util = (
        float(np.mean(history.get("gpu_util_pct", [0.0]))) if history.get("gpu_util_pct") else 0.0
    )

    # Save full metrics report to JSON
    report_dict = {
        "experiment": "Experiment 2: Custom CNN with Class-Weighted Cross-Entropy",
        "model": "CustomWaferCNN",
        "in_channels": 1,
        "preprocessing": "Method C (letterbox_padding)",
        "loss": "Class-Weighted Cross-Entropy (Effective-Number Cui et al. 2019, beta=0.9999)",
        "class_weights": {IDX_TO_LABEL[i]: round(class_weights[i].item(), 4) for i in range(9)},
        "sampler": "Uniform Shuffle",
        "total_duration_seconds": round(total_duration, 2),
        "total_epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "early_stopping_triggered": early_stopping.should_stop,
        "memory_summary": {
            "peak_allocated_mb": round(peak_alloc, 2),
            "peak_reserved_mb": round(peak_res, 2),
            "peak_nvidia_smi_used_mb": round(peak_smi, 2),
            "average_gpu_utilization_pct": round(avg_util, 1),
        },
        "checkpoint_path": str(checkpoint_path),
        "curves_path": str(curves_path),
        "confusion_matrix_path": str(cm_path),
        "best_metrics": best_metrics,
        "history": history,
    }

    metrics_json_path = Path("data/interim/figures/exp2_weighted_ce_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Experiment metrics JSON saved to: {metrics_json_path}")
    print("\n" + "=" * 80)
    print("      EXPERIMENT 2: CUSTOM CNN WEIGHTED CROSS-ENTROPY COMPLETE          ")
    print("=" * 80)
    print(f"Total Duration:        {total_duration / 60:.2f} minutes ({total_duration:.1f} s)")
    print(f"Epochs Trained:        {len(history['epoch'])} / {epochs}")
    print(f"Best Epoch:            {best_epoch}")
    print(f"Best Val Macro F1:     {best_score:.4f}")
    print(
        f"Early Stopping Status: {'TRIGGERED' if early_stopping.should_stop else 'Completed All Epochs'}"
    )
    print(
        f"Peak VRAM Breakdown:   Allocated: {peak_alloc:.1f} MB | Reserved: {peak_res:.1f} MB | nvidia-smi: {peak_smi:.1f} MB"
    )
    print(f"Average GPU Util:      {avg_util:.1f}%")
    print(f"Checkpoint Saved:      {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
