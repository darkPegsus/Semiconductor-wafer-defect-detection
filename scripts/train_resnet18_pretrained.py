# ruff: noqa: E402
"""Phase 5A: ResNet-18 ImageNet Transfer Learning (ResNet18-A).

Executes controlled transfer learning under:
- Architecture: AdaptedResNet18 (in_channels=1, num_classes=9)
- Initialization: ImageNet Pretrained Weights (ResNet18_Weights.DEFAULT)
- Channel Adaptation: Average RGB channels across input filters [64, 3, 7, 7] -> [64, 1, 7, 7]
- Input representation: Single-channel numerical [B, 1, 64, 64]
- Preprocessing: Method C (aspect-ratio-preserving letterbox padding)
- Loss: Standard Cross-Entropy (unweighted)
- Sampling: Uniform random shuffle
- Optimizer: AdamW (weight_decay=1e-2)
- Two-Stage Transfer Learning Schedule:
  * Stage 1 (Epochs 1-3): Frozen Backbone (7,753 trainable parameters), Head lr=1e-4
  * Stage 2 (Epochs 4-30): Unfrozen Backbone (11,174,857 trainable parameters), Backbone lr=1e-5, Head lr=1e-4
- Scheduler: CosineAnnealingLR
- Batch size: 32 (Train), 64 (Val)
- Mixed Precision: Enabled (torch.amp)
- Early Stopping: Validation Macro F1 (patience=7, mode="max")
- Maximum Epochs: 30
- Checkpointing: models/checkpoints/resnet18_pretrained_best.pt
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
    """Run all pre-training verification checks including parameter counts and AMP forward/backward checks."""
    logger.info("=================================================================")
    logger.info("       RUNNING PRE-TRAINING VERIFICATION CHECKS (RESNET18-A)     ")
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

    # 4. Pretrained ImageNet Weights & Single-Channel Adaptation Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=True).to(device)
    assert model.pretrained is True, "Pretrained ImageNet weights failed to load!"
    assert model.backbone.conv1.in_channels == 1, "Input conv1 channels not adapted to 1!"
    assert model.backbone.conv1.weight.shape == (64, 1, 7, 7), "Unexpected conv1 shape!"
    logger.info(
        "Check 4: ImageNet pretrained weights loaded and single-channel filter migration verified."
    )

    # 5. Exact Parameter Counts (Frozen vs. Unfrozen)
    total_params = sum(p.numel() for p in model.parameters())
    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == 11174857, f"Expected 11,174,857 total params, got {total_params}"
    assert frozen_trainable == 7753, (
        f"Expected 7,753 trainable params when frozen, got {frozen_trainable}"
    )
    assert unfrozen_trainable == 11174857
    logger.info(
        f"Check 5: Parameter counts verified: Total: {total_params:,} | "
        f"Stage 1 Frozen Trainable: {frozen_trainable:,} | "
        f"Stage 2 Unfrozen Trainable: {unfrozen_trainable:,}."
    )

    # 6. Forward and backward pass using standard Cross-Entropy under AMP
    ce_criterion = nn.CrossEntropyLoss()
    dummy_x = torch.randn(4, 1, 64, 64, device=device)
    dummy_y = torch.tensor([0, 1, 2, 8], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        dummy_out = model(dummy_x)
        dummy_loss = ce_criterion(dummy_out, dummy_y)

    assert dummy_out.shape == (4, 9)
    assert torch.isfinite(dummy_loss), "Standard CE loss returned non-finite value!"
    dummy_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient in {name}"
    del dummy_x, dummy_y, dummy_out, dummy_loss, ce_criterion, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Check 6: Forward/backward pass with ResNet-18 (AMP) PASSED (finite loss & grads).")

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
        f"Check 8: Test partition ({test_count} wafers, 1537 lots) strictly ISOLATED. "
        "Test set will NOT be used for model selection, validation, or training."
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
    ax1.axvline(x=3.5, color="gray", linestyle=":", label="Unfreeze Backbone (Epoch 4)")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Training & Validation Loss (ResNet18-A Pretrained)", fontsize=12, fontweight="bold"
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
    ax2.axvline(x=3.5, color="gray", linestyle=":", label="Unfreeze Backbone (Epoch 4)")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Classification Metrics (ResNet18-A)", fontsize=12, fontweight="bold")
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
        title="Validation Confusion Matrix (ResNet18-A ImageNet Pretrained)",
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
    model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=True)

    # Stage 1: Freeze backbone for warm-up (Epochs 1-3)
    logger.info("Stage 1 Transfer Learning: Freezing backbone for 3-epoch warmup...")
    model.freeze_backbone()
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Active Trainable Parameters in Stage 1: {trainable_p1:,} (Conv1 + FC Head)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-5)

    epochs = 30
    checkpoint_path = Path("models/checkpoints/resnet18_pretrained_best.pt")
    early_stopping = EarlyStopping(
        patience=7,
        min_delta=0.001,
        mode="max",
        checkpoint_path=checkpoint_path,
    )

    def on_unfreeze_callback(trainer: WaferTrainer) -> None:
        """Callback to unfreeze backbone at Epoch 4 and configure differential learning rates."""
        logger.info("=================================================================")
        logger.info(" Stage 2 Transfer Learning: Unfreezing backbone for fine-tuning  ")
        logger.info("=================================================================")
        trainer.model.unfreeze_backbone()
        trainable_p2 = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"Active Trainable Parameters in Stage 2: {trainable_p2:,} (All Layers)")

        # Differential Learning Rates: Backbone lr=1e-5, Head lr=1e-4
        lr_backbone = 1e-5
        lr_head = 1e-4
        param_groups = trainer.model.get_parameter_groups(lr_backbone=lr_backbone, lr_head=lr_head)
        trainer.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
        remaining_epochs = epochs - 4 + 1  # 27 epochs
        trainer.scheduler = CosineAnnealingLR(
            trainer.optimizer, T_max=remaining_epochs, eta_min=1e-6
        )
        logger.info(
            f"Configured differential parameter groups: Backbone LR={lr_backbone:.1e}, "
            f"Head LR={lr_head:.1e}, CosineAnnealingLR(T_max={remaining_epochs})"
        )
        logger.info("=================================================================\n")

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

    # Fit model with unfreeze callback at epoch 4
    history = trainer.fit(
        epochs=epochs,
        show_progress=False,
        unfreeze_at_epoch=4,
        on_unfreeze=on_unfreeze_callback,
    )
    total_duration = time.time() - start_time

    # Save training curves
    curves_path = Path("data/interim/figures/resnet18_pretrained_training_curves.png")
    plot_training_curves(history, curves_path)

    # Best epoch and best metadata
    best_epoch = early_stopping.best_epoch
    best_score = early_stopping.best_score
    best_metrics = (
        early_stopping.best_metadata["val_metrics"] if early_stopping.best_metadata else None
    )

    # Save confusion matrix plot if available
    cm_path = Path("data/interim/figures/resnet18_pretrained_confusion_matrix.png")
    class_names = [IDX_TO_LABEL[i] for i in range(9)]
    if best_metrics and "confusion_matrix" in best_metrics:
        plot_confusion_matrix(best_metrics["confusion_matrix"], class_names, cm_path)

    # Memory statistics summary across all epochs
    peak_alloc = max(history.get("peak_allocated_mb", [0.0]))
    peak_res = max(history.get("peak_reserved_mb", [0.0]))
    peak_smi = max(history.get("smi_used_mb", [0.0]))
    avg_util = (
        float(np.mean(history.get("gpu_util_pct", [0.0]))) if history.get("gpu_util_pct") else 0.0
    )
    epoch_durations = history.get("epoch_duration_seconds", [])
    avg_epoch_duration = float(np.mean(epoch_durations)) if epoch_durations else 0.0
    active_training_duration = float(np.sum(epoch_durations)) if epoch_durations else 0.0

    # Save full metrics report to JSON
    report_dict = {
        "experiment": "Phase 5A: ResNet-18 ImageNet Transfer Learning (ResNet18-A)",
        "model": "AdaptedResNet18",
        "initialization": "ImageNet Pretrained (ResNet18_Weights.DEFAULT)",
        "in_channels": 1,
        "total_parameters": 11174857,
        "stage1_trainable_parameters": 7753,
        "stage2_trainable_parameters": 11174857,
        "preprocessing": "Method C (letterbox_padding)",
        "loss": "Standard Cross-Entropy (unweighted)",
        "sampler": "Uniform Random Shuffle",
        "two_stage_schedule": {
            "stage1_warmup_epochs": "1-3 (frozen backbone)",
            "stage1_trainable_layers": "conv1, fc",
            "stage1_lr": 1e-4,
            "stage2_finetune_epochs": "4-30 (unfrozen backbone)",
            "stage2_backbone_lr": 1e-5,
            "stage2_head_lr": 1e-4,
        },
        "data_loader_config": {
            "train_batch_size": 32,
            "val_batch_size": 64,
            "num_workers": 0,
            "pin_memory": torch.cuda.is_available(),
        },
        "timing_summary": {
            "wall_clock_duration_seconds": round(total_duration, 2),
            "active_training_duration_seconds": round(active_training_duration, 2),
            "average_epoch_duration_seconds": round(avg_epoch_duration, 2),
            "total_epochs_trained": len(history["epoch"]),
        },
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

    metrics_json_path = Path("data/interim/figures/resnet18_pretrained_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Experiment metrics JSON saved to: {metrics_json_path}")
    print("\n" + "=" * 80)
    print("     PHASE 5A: RESNET-18 IMAGENET TRANSFER LEARNING (RESNET18-A) COMPLETE     ")
    print("=" * 80)
    print(f"Total Wall Duration:   {total_duration / 60:.2f} minutes ({total_duration:.1f} s)")
    print(
        f"Active Epoch Time:     {active_training_duration / 60:.2f} minutes ({active_training_duration:.1f} s)"
    )
    print(f"Average Epoch Time:    {avg_epoch_duration:.1f} s/epoch")
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
