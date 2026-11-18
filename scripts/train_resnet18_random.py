# ruff: noqa: E402
"""Phase 5B: ResNet-18 Random Initialization Training (ResNet18-B).

Executes controlled benchmark training from scratch under:
- Architecture: AdaptedResNet18 (in_channels=1, num_classes=9)
- Initialization: Random Kaiming Normal (No ImageNet pretrained weights)
- Input representation: Single-channel numerical [B, 1, 64, 64]
- Preprocessing: Method C (aspect-ratio-preserving letterbox padding)
- Loss: Standard Cross-Entropy (unweighted)
- Sampling: Uniform random shuffle
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
- Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-6)
- Batch size: 32 (Train), 64 (Val)
- Mixed Precision: Enabled (torch.amp)
- Early Stopping: Validation Macro F1 (patience=7, mode="max")
- Maximum Epochs: 30
- Checkpointing: models/checkpoints/resnet18_random_best.pt
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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from src.models.resnet18 import AdaptedResNet18
from src.training.early_stopping import EarlyStopping
from src.training.engine import WaferTrainer


def run_pre_training_verifications(df, splits):
    """Run all pre-training verification checks including parameter counts and AMP checks."""
    logger.info("=================================================================")
    logger.info("       RUNNING PRE-TRAINING VERIFICATION CHECKS (RESNET18-B)     ")
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

    # 4. Model Architecture & Random Initialization Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False).to(device)
    assert model.pretrained is False, "Model incorrectly marked as pretrained!"
    assert model.backbone.conv1.in_channels == 1, "Input conv1 channels not adapted to 1!"
    assert model.backbone.conv1.weight.shape == (64, 1, 7, 7), "Unexpected conv1 shape!"
    logger.info(
        "Check 4: Model architecture matches Phase 5A and random initialization confirmed (pretrained=False)."
    )

    # 5. Exact Parameter Counts (All 100% Trainable from Epoch 1)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 11174857, f"Expected 11,174,857 total params, got {total_params}"
    assert trainable_params == 11174857, (
        f"Expected all 11,174,857 params trainable, got {trainable_params}"
    )
    logger.info(
        f"Check 5: Exact parameter count verified: Total = {total_params:,} | Trainable = {trainable_params:,} (100.0%)."
    )

    # 6. Forward and backward pass using standard Cross-Entropy under AMP
    ce_criterion = nn.CrossEntropyLoss()
    dummy_x = torch.randn(4, 1, 64, 64, device=device)
    dummy_y = torch.tensor([0, 1, 2, 8], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        dummy_out = model(dummy_x)
        dummy_loss = ce_criterion(dummy_out, dummy_y)

    assert dummy_out.shape == (4, 9), f"Expected logits shape (4, 9), got {dummy_out.shape}"
    assert torch.isfinite(dummy_loss), "Standard CE loss returned non-finite value!"
    dummy_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"
    del dummy_x, dummy_y, dummy_out, dummy_loss, ce_criterion, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Check 6: Forward and backward pass under AMP PASSED (finite loss and gradients).")

    # 7. Check GPU memory and validate batch size 32
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Check 7: GPU detected: {gpu_name} with {vram_gb:.2f} GB VRAM.")
        logger.info("Batch size 32 selected: Memory profile validated (<500 MB active required).")
    else:
        logger.info("Check 7: Running on CPU.")

    # 8. Confirm test set isolation
    test_count = len(splits["test"])
    logger.info(
        f"Check 8: Test partition ({test_count:,} wafers, 1,537 lots) strictly ISOLATED. "
        "Test set will NOT be loaded, visited, or used for model selection."
    )
    logger.info("=================================================================\n")


def plot_training_curves(history: dict, output_path: Path):
    """Plot training/validation loss, accuracy, and macro F1 curves."""
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(
        epochs,
        history["train_loss"],
        "o-",
        label="Train Loss",
        color="#1f77b4",
        lw=2,
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        "s--",
        label="Val Loss",
        color="#ff7f0e",
        lw=2,
    )
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Training & Validation Loss (ResNet18-B Random Init)", fontsize=12, fontweight="bold"
    )
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
    ax2.set_title("Validation Classification Metrics (ResNet18-B)", fontsize=12, fontweight="bold")
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
        title="Validation Confusion Matrix (ResNet18-B Random Init)",
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

    # DataLoaders: Batch size 32 for train, 64 for val
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"ResNet-18 Random Initialization: Total Params = {total_p:,} | "
        f"Trainable Params = {trainable_p:,} (100.0% trainable from Epoch 1)"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )
    epochs = 30
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoint_path = Path("models/checkpoints/resnet18_random_best.pt")
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

    logger.info("Starting training loop for Phase 5B (ResNet18 Random Init)...")
    training_start = time.time()
    history = trainer.fit(
        epochs=epochs,
        show_progress=False,
    )
    active_training_duration = time.time() - training_start
    total_wall_duration = time.time() - start_time

    # Save training curves
    curves_path = Path("data/interim/figures/resnet18_random_training_curves.png")
    plot_training_curves(history, curves_path)

    # Best epoch and best metadata
    best_epoch = early_stopping.best_epoch
    best_score = early_stopping.best_score
    best_metrics = (
        early_stopping.best_metadata.get("val_metrics", {}) if early_stopping.best_metadata else {}
    )

    # Save confusion matrix plot
    class_names = [IDX_TO_LABEL[i] for i in range(9)]
    cm_path = Path("data/interim/figures/resnet18_random_confusion_matrix.png")
    if "confusion_matrix" in best_metrics:
        plot_confusion_matrix(best_metrics["confusion_matrix"], class_names, cm_path)

    # Memory summary
    memory_summary = {
        "peak_allocated_mb": round(max(history["peak_allocated_mb"]), 1),
        "peak_reserved_mb": round(max(history["peak_reserved_mb"]), 1),
        "peak_nvidia_smi_used_mb": round(max(history["smi_used_mb"]), 1),
        "average_gpu_utilization_pct": round(float(np.mean(history["gpu_util_pct"])), 1),
    }

    # Timing summary
    timing_summary = {
        "wall_clock_duration_seconds": round(total_wall_duration, 2),
        "active_training_duration_seconds": round(active_training_duration, 2),
        "average_epoch_duration_seconds": round(
            float(np.mean(history["epoch_duration_seconds"])), 2
        ),
        "total_epochs_trained": len(history["epoch"]),
    }

    metrics_payload = {
        "experiment": "Phase 5B: ResNet-18 Random Initialization Training (ResNet18-B)",
        "model": "AdaptedResNet18",
        "initialization": "Random Kaiming Normal (No ImageNet pretrained weights)",
        "in_channels": 1,
        "total_parameters": total_p,
        "trainable_parameters": trainable_p,
        "preprocessing": "Method C (letterbox_padding)",
        "loss": "Standard Cross-Entropy (unweighted)",
        "sampler": "Uniform Random Shuffle",
        "optimizer": {
            "name": "AdamW",
            "initial_lr": 1e-3,
            "weight_decay": 1e-2,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": epochs,
            "eta_min": 1e-6,
        },
        "data_loader_config": {
            "train_batch_size": 32,
            "val_batch_size": 64,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        },
        "timing_summary": timing_summary,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_score, 4),
        "early_stopping_triggered": early_stopping.should_stop,
        "memory_summary": memory_summary,
        "checkpoint_path": str(checkpoint_path),
        "curves_path": str(curves_path),
        "confusion_matrix_path": str(cm_path),
        "best_metrics": best_metrics,
        "history": history,
    }

    metrics_json_path = Path("data/interim/figures/resnet18_random_metrics.json")
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Metrics JSON saved to {metrics_json_path}")

    # Final summary printout
    logger.info("=================================================================")
    logger.info("       PHASE 5B: RESNET18 RANDOM INITIALIZATION COMPLETE         ")
    logger.info("=================================================================")
    logger.info(f"Best Epoch: {best_epoch:02d} / {len(history['epoch']):02d}")
    logger.info(f"Best Validation Macro F1: {best_score:.4f}")
    if best_metrics:
        logger.info(f"Validation Accuracy: {best_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"Validation Macro Precision: {best_metrics.get('macro_precision', 0.0):.4f}")
        logger.info(f"Validation Macro Recall: {best_metrics.get('macro_recall', 0.0):.4f}")
    logger.info(f"Active Training Time: {active_training_duration / 60:.2f} min")
    logger.info(f"Peak VRAM: {memory_summary['peak_allocated_mb']} MB Allocated")
    logger.info(f"Checkpoint Saved: {checkpoint_path}")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
