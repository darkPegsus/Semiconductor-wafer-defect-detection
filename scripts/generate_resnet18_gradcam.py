# ruff: noqa: E402
"""Phase 6B: Generate Grad-CAM Visualizations for AdaptedResNet18 and Controlled Architectural Comparison.

Loads both winning checkpoints:
1. Pretrained AdaptedResNet18 (models/checkpoints/resnet18_pretrained_best.pt, Best Val Macro F1 = 0.8611)
2. CustomWaferCNN Baseline (models/checkpoints/exp1_custom_cnn_baseline_best.pt, Best Val Macro F1 = 0.8861)

Executes Grad-CAM targeting:
- AdaptedResNet18: 'layer4' (model.backbone.layer4, feature map 2x2 before pooling)
- CustomWaferCNN: 'features.conv_block4' (model.block4, feature map 4x4 before pooling)

Generates:
1. ResNet-18 specific visualizations in data/interim/figures/gradcam/resnet18/
2. Side-by-side comparative figures in data/interim/figures/gradcam/comparison/
3. 9-class comparative grid illustrating spatial focus differences between architectures

Strictly uses the validation partition (N_val = 24,705). The test set remains untouched.
"""

import os
import sys
import tempfile
from pathlib import Path

# Fix Matplotlib cache warning on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from src.data.dataset import (
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WM811kDataset,
    load_labeled_dataset,
)
from src.data.group_split import load_split_indices
from src.explainability.grad_cam import GradCAM, overlay_heatmap_on_wafer
from src.models.custom_cnn import CustomWaferCNN
from src.models.resnet18 import AdaptedResNet18


def plot_resnet18_single(
    wafer_map: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    gt_label: str,
    pred_label: str,
    confidence: float,
    case_title: str,
    output_path: Path,
    target_class_label: str | None = None,
) -> None:
    """Render and save a 3-panel Grad-CAM figure for AdaptedResNet18."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Discrete wafer rendering
    h, w = wafer_map.shape
    base_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    base_rgb[wafer_map == 0] = [28, 32, 42]
    base_rgb[wafer_map == 1] = [170, 175, 185]
    base_rgb[wafer_map == 2] = [235, 45, 45]

    # Panel 1: Original Wafer Map
    axes[0].imshow(base_rgb)
    axes[0].set_title(
        f"Original Wafer Map (64x64)\nGround Truth: {gt_label}", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    target_str = f"Target: {target_class_label}" if target_class_label else f"Target: {pred_label}"
    axes[1].set_title(f"ResNet-18 CAM (layer4 [2x2])\n{target_str}", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    status_str = "CORRECT" if gt_label == pred_label else "MISCLASSIFIED"
    status_color = "green" if gt_label == pred_label else "red"
    axes[2].set_title(
        f"Die-Preserving Overlay\nPred: {pred_label} ({confidence * 100:.1f}%) [{status_str}]",
        fontsize=11,
        fontweight="bold",
        color=status_color,
    )
    axes[2].axis("off")

    fig.suptitle(case_title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_comparison(
    wafer_map: np.ndarray,
    cam_cnn: np.ndarray,
    overlay_cnn: np.ndarray,
    cnn_pred_label: str,
    cnn_conf: float,
    cam_resnet: np.ndarray,
    overlay_resnet: np.ndarray,
    resnet_pred_label: str,
    resnet_conf: float,
    gt_label: str,
    case_title: str,
    output_path: Path,
) -> None:
    """Render and save a 5-panel side-by-side architecture comparison figure.

    Panels:
    1. Original Wafer Map (64x64)
    2. CustomWaferCNN Grad-CAM Heatmap (conv_block4, 4x4 feature grid)
    3. CustomWaferCNN Overlay
    4. AdaptedResNet18 Grad-CAM Heatmap (layer4, 2x2 feature grid)
    5. AdaptedResNet18 Overlay
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    # Discrete wafer rendering
    h, w = wafer_map.shape
    base_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    base_rgb[wafer_map == 0] = [28, 32, 42]
    base_rgb[wafer_map == 1] = [170, 175, 185]
    base_rgb[wafer_map == 2] = [235, 45, 45]

    # Panel 1: Original Wafer Map
    axes[0].imshow(base_rgb)
    axes[0].set_title(
        f"Original Wafer (64x64)\nGround Truth: {gt_label}", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: Custom CNN Heatmap
    axes[1].imshow(cam_cnn, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        "Custom CNN CAM\n(conv_block4 [4x4])", fontsize=11, fontweight="bold", color="navy"
    )
    axes[1].axis("off")

    # Panel 3: Custom CNN Overlay
    axes[2].imshow(overlay_cnn)
    cnn_status = "CORRECT" if cnn_pred_label == gt_label else "MISSED"
    cnn_color = "darkgreen" if cnn_pred_label == gt_label else "crimson"
    axes[2].set_title(
        f"Custom CNN Overlay\nPred: {cnn_pred_label} ({cnn_conf * 100:.1f}%)\n[{cnn_status}]",
        fontsize=10,
        fontweight="bold",
        color=cnn_color,
    )
    axes[2].axis("off")

    # Panel 4: ResNet18 Heatmap
    axes[3].imshow(cam_resnet, cmap="jet", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        "ResNet-18 CAM\n(layer4 [2x2])", fontsize=11, fontweight="bold", color="darkred"
    )
    axes[3].axis("off")

    # Panel 5: ResNet18 Overlay
    axes[4].imshow(overlay_resnet)
    resnet_status = "CORRECT" if resnet_pred_label == gt_label else "MISSED"
    resnet_color = "darkgreen" if resnet_pred_label == gt_label else "crimson"
    axes[4].set_title(
        f"ResNet-18 Overlay\nPred: {resnet_pred_label} ({resnet_conf * 100:.1f}%)\n[{resnet_status}]",
        fontsize=10,
        fontweight="bold",
        color=resnet_color,
    )
    axes[4].axis("off")

    fig.suptitle(case_title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    logger.info("=================================================================")
    logger.info("  PHASE 6B: RESNET-18 GRAD-CAM & ARCHITECTURAL COMPARISON        ")
    logger.info("=================================================================")

    # 1. Load dataset & validation split
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")

    val_indices = splits["val"]
    test_indices = splits["test"]

    # Strict isolation check
    assert len(test_indices) == 24705, f"Expected 24705 test samples, got {len(test_indices)}"
    logger.info("Verified: Test partition strictly unvisited (24,705 held-out samples).")

    val_df = df.iloc[val_indices].reset_index(drop=True)
    val_dataset = WM811kDataset(
        val_df,
        preprocessing_method="method_c",
        representation="single_channel",
    )
    logger.info(f"Loaded validation dataset: {len(val_dataset):,} wafers.")

    # 2. Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load ResNet-18 Pretrained Checkpoint
    resnet_path = PROJECT_ROOT / "models" / "checkpoints" / "resnet18_pretrained_best.pt"
    if not resnet_path.exists():
        raise FileNotFoundError(f"ResNet-18 checkpoint not found at {resnet_path}")
    model_resnet = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False).to(device)
    ckpt_resnet = torch.load(resnet_path, map_location=device)
    model_resnet.load_state_dict(ckpt_resnet["model_state_dict"])
    model_resnet.eval()
    logger.info(f"Loaded ResNet-18 checkpoint: Best Val Macro F1 = {ckpt_resnet.get('best_score')}")

    # Load CustomWaferCNN Baseline Checkpoint
    cnn_path = PROJECT_ROOT / "models" / "checkpoints" / "exp1_custom_cnn_baseline_best.pt"
    if not cnn_path.exists():
        raise FileNotFoundError(f"Custom CNN checkpoint not found at {cnn_path}")
    model_cnn = CustomWaferCNN(in_channels=1, num_classes=9).to(device)
    ckpt_cnn = torch.load(cnn_path, map_location=device)
    model_cnn.load_state_dict(ckpt_cnn["model_state_dict"])
    model_cnn.eval()
    logger.info(f"Loaded Custom CNN checkpoint: Best Val Macro F1 = {ckpt_cnn.get('best_score')}")

    # 3. Setup GradCAM engines
    cam_resnet = GradCAM(model=model_resnet, target_layer="layer4", device=device)
    cam_cnn = GradCAM(model=model_cnn, target_layer="features.conv_block4", device=device)
    logger.info("Initialized GradCAM engines: ResNet-18 (layer4) and Custom CNN (conv_block4).")

    # Output directories
    resnet_dir = PROJECT_ROOT / "data" / "interim" / "figures" / "gradcam" / "resnet18"
    compare_dir = PROJECT_ROOT / "data" / "interim" / "figures" / "gradcam" / "comparison"
    resnet_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # 4. Search validation set for representative cases
    logger.info("Scanning validation set across both models...")

    # Tracking dictionaries / lists
    both_correct = {}  # class_idx -> list of (idx, cnn_conf, resnet_conf)
    cnn_correct_resnet_missed = []  # list of (idx, gt, cnn_pred, cnn_conf, resnet_pred, resnet_conf)
    resnet_correct_cnn_missed = []  # list of (idx, gt, cnn_pred, cnn_conf, resnet_pred, resnet_conf)
    false_alarms_resnet = []  # GT=0, ResNet!=0: (idx, gt, resnet_pred, resnet_conf)
    confusions_resnet = []  # GT!=0, ResNet!=0, GT!=ResNet: (idx, gt, resnet_pred, resnet_conf)
    low_conf_resnet = []  # conf in [0.35, 0.65]

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_x = batch["image"].to(device)
            batch_y = batch["label"]

            # ResNet-18 inference
            logits_resnet = model_resnet(batch_x)
            probs_resnet = torch.softmax(logits_resnet, dim=1)
            preds_resnet = torch.argmax(logits_resnet, dim=1)

            # Custom CNN inference
            logits_cnn = model_cnn(batch_x)
            probs_cnn = torch.softmax(logits_cnn, dim=1)
            preds_cnn = torch.argmax(logits_cnn, dim=1)

            for b in range(len(batch_y)):
                idx = sample_offset + b
                gt = int(batch_y[b].item())

                p_res = int(preds_resnet[b].item())
                c_res = float(probs_resnet[b, p_res].item())

                p_cnn = int(preds_cnn[b].item())
                c_cnn = float(probs_cnn[b, p_cnn].item())

                # Both correct
                if gt == p_res and gt == p_cnn:
                    if gt not in both_correct:
                        both_correct[gt] = []
                    both_correct[gt].append((idx, c_cnn, c_res))

                # Custom CNN correct, ResNet-18 missed (defect escape)
                if gt == p_cnn and gt != p_res:
                    cnn_correct_resnet_missed.append((idx, gt, p_cnn, c_cnn, p_res, c_res))

                # ResNet-18 correct, Custom CNN missed
                if gt == p_res and gt != p_cnn:
                    resnet_correct_cnn_missed.append((idx, gt, p_cnn, c_cnn, p_res, c_res))

                # ResNet-18 false alarm on clean wafer
                if gt == 0 and p_res != 0 and len(false_alarms_resnet) < 10:
                    false_alarms_resnet.append((idx, gt, p_res, c_res))

                # ResNet-18 defect confusion
                if gt != 0 and p_res != 0 and gt != p_res and len(confusions_resnet) < 10:
                    confusions_resnet.append((idx, gt, p_res, c_res))

                # ResNet-18 low confidence
                if 0.35 <= c_res <= 0.65 and len(low_conf_resnet) < 10:
                    low_conf_resnet.append((idx, gt, p_res, c_res))

            sample_offset += len(batch_y)

    logger.info(f"Discovered: {len(both_correct)} / 9 classes with both models correct.")
    logger.info(
        f"Discovered: {len(cnn_correct_resnet_missed)} cases where Custom CNN is correct and ResNet-18 missed."
    )
    logger.info(
        f"Discovered: {len(resnet_correct_cnn_missed)} cases where ResNet-18 is correct and Custom CNN missed."
    )
    logger.info(f"Discovered: {len(false_alarms_resnet)} ResNet-18 false alarms on clean wafers.")

    # -------------------------------------------------------------
    # 5. Generate ResNet-18 Specific Visualizations
    # -------------------------------------------------------------
    logger.info("Generating ResNet-18 specific visualizations...")
    saved_resnet_paths = []

    # 5A. Correct samples across all 9 classes for ResNet-18
    for cls_idx in sorted(both_correct.keys()):
        # Select sample with highest ResNet confidence
        candidates = sorted(both_correct[cls_idx], key=lambda x: x[2], reverse=True)
        val_idx, _, res_conf = candidates[0]

        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        cam_map, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=cls_idx)
        heatmap_np = cam_map.squeeze().numpy()
        overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

        cls_name = IDX_TO_LABEL[cls_idx]
        filename = f"resnet18_correct_{cls_idx}_{cls_name.lower().replace('-', '_')}.png"
        out_path = resnet_dir / filename
        plot_resnet18_single(
            wafer_map=wafer_np,
            heatmap=heatmap_np,
            overlay=overlay_np,
            gt_label=cls_name,
            pred_label=cls_name,
            confidence=res_conf,
            case_title=f"AdaptedResNet18 Correct: {cls_name} (Confidence: {res_conf * 100:.1f}%)",
            output_path=out_path,
        )
        saved_resnet_paths.append(out_path)

    # 5B. ResNet-18 Missed Defects (Scratch, Loc, Edge-Loc predicted as none)
    target_defect_misses = [
        LABEL_TO_IDX["Scratch"],
        LABEL_TO_IDX["Loc"],
        LABEL_TO_IDX["Edge-Loc"],
    ]
    for target_gt in target_defect_misses:
        matching = [
            item
            for item in cnn_correct_resnet_missed
            if item[1] == target_gt and item[4] == LABEL_TO_IDX["none"]
        ]
        if matching:
            val_idx, gt, _, _, p_res, c_res = matching[0]
            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)
            gt_name = IDX_TO_LABEL[gt]
            pred_name = IDX_TO_LABEL[p_res]

            # Attribution to predicted class ('none')
            cam_pred, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_res)
            heatmap_pred = cam_pred.squeeze().numpy()
            overlay_pred = overlay_heatmap_on_wafer(
                wafer_np, heatmap_pred, alpha=0.5, colormap="jet"
            )

            out_pred = resnet_dir / f"resnet18_missed_{gt_name.lower()}_pred_none.png"
            plot_resnet18_single(
                wafer_map=wafer_np,
                heatmap=heatmap_pred,
                overlay=overlay_pred,
                gt_label=gt_name,
                pred_label=pred_name,
                confidence=c_res,
                case_title=f"ResNet-18 Missed Defect: True {gt_name} predicted as {pred_name} ({c_res * 100:.1f}%)",
                output_path=out_pred,
            )
            saved_resnet_paths.append(out_pred)

            # Attribution to true class
            cam_gt, _, prob_gt = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=gt)
            heatmap_gt = cam_gt.squeeze().numpy()
            overlay_gt = overlay_heatmap_on_wafer(wafer_np, heatmap_gt, alpha=0.5, colormap="jet")
            gt_conf = float(prob_gt[0, gt].item())

            out_gt = resnet_dir / f"resnet18_missed_{gt_name.lower()}_target_true_class.png"
            plot_resnet18_single(
                wafer_map=wafer_np,
                heatmap=heatmap_gt,
                overlay=overlay_gt,
                gt_label=gt_name,
                pred_label=pred_name,
                confidence=gt_conf,
                case_title=f"ResNet-18 Attribution to True Class ({gt_name}) for Missed Sample",
                output_path=out_gt,
                target_class_label=gt_name,
            )
            saved_resnet_paths.append(out_gt)

    # 5C. ResNet-18 False Alarms on Clean Wafers
    for i, (val_idx, gt, p_res, c_res) in enumerate(false_alarms_resnet[:2]):
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)
        gt_name = IDX_TO_LABEL[gt]
        pred_name = IDX_TO_LABEL[p_res]

        cam_fa, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_res)
        heatmap_fa = cam_fa.squeeze().numpy()
        overlay_fa = overlay_heatmap_on_wafer(wafer_np, heatmap_fa, alpha=0.5, colormap="jet")

        out_fa = resnet_dir / f"resnet18_false_alarm_{i + 1}_{pred_name.lower()}.png"
        plot_resnet18_single(
            wafer_map=wafer_np,
            heatmap=heatmap_fa,
            overlay=overlay_fa,
            gt_label=gt_name,
            pred_label=pred_name,
            confidence=c_res,
            case_title=f"ResNet-18 False Alarm: Clean Wafer Falsely Classified as {pred_name}",
            output_path=out_fa,
        )
        saved_resnet_paths.append(out_fa)

    # 5D. ResNet-18 9-Class Summary Grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(9):
        if i in both_correct:
            val_idx, _, res_conf = sorted(both_correct[i], key=lambda x: x[2], reverse=True)[0]
            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

            cam_map, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=i)
            heatmap_np = cam_map.squeeze().numpy()
            overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

            axes[i].imshow(overlay_np)
            cls_name = IDX_TO_LABEL[i]
            axes[i].set_title(
                f"{i}: {cls_name}\nConf: {res_conf * 100:.1f}%",
                fontsize=11,
                fontweight="bold",
            )
            axes[i].axis("off")

    plt.suptitle(
        "AdaptedResNet18 Grad-CAM Visualizations across All 9 Defect Classes (layer4)",
        fontsize=15,
        fontweight="bold",
        y=0.94,
    )
    plt.tight_layout()
    grid_resnet = resnet_dir / "resnet18_summary_9classes.png"
    fig.savefig(grid_resnet, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_resnet_paths.append(grid_resnet)

    logger.info(f"Saved {len(saved_resnet_paths)} ResNet-18 specific figures to {resnet_dir}.")

    # -------------------------------------------------------------
    # 6. Generate Controlled Architectural Comparisons (5 Panels)
    # -------------------------------------------------------------
    logger.info("Generating Side-by-Side Architectural Comparisons (Custom CNN vs ResNet-18)...")
    saved_compare_paths = []

    # 6A. Controlled Comparison on Correct Defect Classes
    # Show comparison across representative classes: Center, Donut, Edge-Ring, Loc, Scratch
    rep_classes = [
        LABEL_TO_IDX["Center"],
        LABEL_TO_IDX["Donut"],
        LABEL_TO_IDX["Edge-Ring"],
        LABEL_TO_IDX["Loc"],
        LABEL_TO_IDX["Scratch"],
        LABEL_TO_IDX["none"],
    ]
    for cls_idx in rep_classes:
        if cls_idx in both_correct:
            # Pick a sample where both are highly confident
            candidates = sorted(both_correct[cls_idx], key=lambda x: x[1] + x[2], reverse=True)
            val_idx, cnn_conf, res_conf = candidates[0]

            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)
            cls_name = IDX_TO_LABEL[cls_idx]

            # Custom CNN CAM
            cam_c, _, _ = cam_cnn.generate_cam(wafer_tensor.unsqueeze(0), target_class=cls_idx)
            cam_c_np = cam_c.squeeze().numpy()
            overlay_c_np = overlay_heatmap_on_wafer(wafer_np, cam_c_np, alpha=0.5, colormap="jet")

            # ResNet-18 CAM
            cam_r, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=cls_idx)
            cam_r_np = cam_r.squeeze().numpy()
            overlay_r_np = overlay_heatmap_on_wafer(wafer_np, cam_r_np, alpha=0.5, colormap="jet")

            out_cmp = (
                compare_dir
                / f"comparison_correct_{cls_idx}_{cls_name.lower().replace('-', '_')}.png"
            )
            plot_side_by_side_comparison(
                wafer_map=wafer_np,
                cam_cnn=cam_c_np,
                overlay_cnn=overlay_c_np,
                cnn_pred_label=cls_name,
                cnn_conf=cnn_conf,
                cam_resnet=cam_r_np,
                overlay_resnet=overlay_r_np,
                resnet_pred_label=cls_name,
                resnet_conf=res_conf,
                gt_label=cls_name,
                case_title=f"Architecture Comparison: Correct {cls_name} Classification",
                output_path=out_cmp,
            )
            saved_compare_paths.append(out_cmp)

    # 6B. Discrepancy: Custom CNN Detects Scratch, ResNet-18 Misses (Predicts 'none')
    scratch_misses = [
        item
        for item in cnn_correct_resnet_missed
        if item[1] == LABEL_TO_IDX["Scratch"] and item[4] == LABEL_TO_IDX["none"]
    ]
    if scratch_misses:
        val_idx, gt, p_cnn, c_cnn, p_res, c_res = scratch_misses[0]
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        cam_c, _, _ = cam_cnn.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_cnn)
        cam_c_np = cam_c.squeeze().numpy()
        overlay_c_np = overlay_heatmap_on_wafer(wafer_np, cam_c_np, alpha=0.5, colormap="jet")

        cam_r, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_res)
        cam_r_np = cam_r.squeeze().numpy()
        overlay_r_np = overlay_heatmap_on_wafer(wafer_np, cam_r_np, alpha=0.5, colormap="jet")

        out_scratch = compare_dir / "comparison_discrepancy_scratch_missed_by_resnet18.png"
        plot_side_by_side_comparison(
            wafer_map=wafer_np,
            cam_cnn=cam_c_np,
            overlay_cnn=overlay_c_np,
            cnn_pred_label=IDX_TO_LABEL[p_cnn],
            cnn_conf=c_cnn,
            cam_resnet=cam_r_np,
            overlay_resnet=overlay_r_np,
            resnet_pred_label=IDX_TO_LABEL[p_res],
            resnet_conf=c_res,
            gt_label="Scratch",
            case_title="Discrepancy: Custom CNN Detects Thin Scratch vs. ResNet-18 Escape (Pred 'none')",
            output_path=out_scratch,
        )
        saved_compare_paths.append(out_scratch)

    # 6C. Discrepancy: Custom CNN Detects Loc, ResNet-18 Misses (Predicts 'none')
    loc_misses = [
        item
        for item in cnn_correct_resnet_missed
        if item[1] == LABEL_TO_IDX["Loc"] and item[4] == LABEL_TO_IDX["none"]
    ]
    if loc_misses:
        val_idx, gt, p_cnn, c_cnn, p_res, c_res = loc_misses[0]
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        cam_c, _, _ = cam_cnn.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_cnn)
        cam_c_np = cam_c.squeeze().numpy()
        overlay_c_np = overlay_heatmap_on_wafer(wafer_np, cam_c_np, alpha=0.5, colormap="jet")

        cam_r, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_res)
        cam_r_np = cam_r.squeeze().numpy()
        overlay_r_np = overlay_heatmap_on_wafer(wafer_np, cam_r_np, alpha=0.5, colormap="jet")

        out_loc = compare_dir / "comparison_discrepancy_loc_missed_by_resnet18.png"
        plot_side_by_side_comparison(
            wafer_map=wafer_np,
            cam_cnn=cam_c_np,
            overlay_cnn=overlay_c_np,
            cnn_pred_label=IDX_TO_LABEL[p_cnn],
            cnn_conf=c_cnn,
            cam_resnet=cam_r_np,
            overlay_resnet=overlay_r_np,
            resnet_pred_label=IDX_TO_LABEL[p_res],
            resnet_conf=c_res,
            gt_label="Loc",
            case_title="Discrepancy: Custom CNN Detects Small Loc vs. ResNet-18 Escape (Pred 'none')",
            output_path=out_loc,
        )
        saved_compare_paths.append(out_loc)

    # 6D. Discrepancy: Edge-Loc Defect Comparison
    edgeloc_misses = [
        item for item in cnn_correct_resnet_missed if item[1] == LABEL_TO_IDX["Edge-Loc"]
    ]
    if edgeloc_misses:
        val_idx, gt, p_cnn, c_cnn, p_res, c_res = edgeloc_misses[0]
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        cam_c, _, _ = cam_cnn.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_cnn)
        cam_c_np = cam_c.squeeze().numpy()
        overlay_c_np = overlay_heatmap_on_wafer(wafer_np, cam_c_np, alpha=0.5, colormap="jet")

        cam_r, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=p_res)
        cam_r_np = cam_r.squeeze().numpy()
        overlay_r_np = overlay_heatmap_on_wafer(wafer_np, cam_r_np, alpha=0.5, colormap="jet")

        out_edgeloc = compare_dir / "comparison_discrepancy_edgeloc.png"
        plot_side_by_side_comparison(
            wafer_map=wafer_np,
            cam_cnn=cam_c_np,
            overlay_cnn=overlay_c_np,
            cnn_pred_label=IDX_TO_LABEL[p_cnn],
            cnn_conf=c_cnn,
            cam_resnet=cam_r_np,
            overlay_resnet=overlay_r_np,
            resnet_pred_label=IDX_TO_LABEL[p_res],
            resnet_conf=c_res,
            gt_label="Edge-Loc",
            case_title=f"Discrepancy: Edge-Loc Attributions (Custom CNN {c_cnn * 100:.1f}% vs ResNet-18 {c_res * 100:.1f}%)",
            output_path=out_edgeloc,
        )
        saved_compare_paths.append(out_edgeloc)

    # 6E. Composite 9-Class Comparison Grid
    logger.info("Generating 9-class composite architecture comparison grid...")
    fig, axes = plt.subplots(9, 3, figsize=(12, 32))

    for i in range(9):
        cls_name = IDX_TO_LABEL[i]
        if i in both_correct:
            val_idx, cnn_conf, res_conf = sorted(
                both_correct[i], key=lambda x: x[1] + x[2], reverse=True
            )[0]
            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

            # Discrete wafer rendering
            h, w = wafer_np.shape
            base_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            base_rgb[wafer_np == 0] = [28, 32, 42]
            base_rgb[wafer_np == 1] = [170, 175, 185]
            base_rgb[wafer_np == 2] = [235, 45, 45]

            # Custom CNN
            cam_c, _, _ = cam_cnn.generate_cam(wafer_tensor.unsqueeze(0), target_class=i)
            overlay_c = overlay_heatmap_on_wafer(
                wafer_np, cam_c.squeeze().numpy(), alpha=0.5, colormap="jet"
            )

            # ResNet-18
            cam_r, _, _ = cam_resnet.generate_cam(wafer_tensor.unsqueeze(0), target_class=i)
            overlay_r = overlay_heatmap_on_wafer(
                wafer_np, cam_r.squeeze().numpy(), alpha=0.5, colormap="jet"
            )

            # Col 0: Original Wafer
            axes[i, 0].imshow(base_rgb)
            axes[i, 0].set_title(f"Class {i}: {cls_name}", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            # Col 1: Custom CNN Overlay
            axes[i, 1].imshow(overlay_c)
            axes[i, 1].set_title(
                f"Custom CNN (4x4)\nConf: {cnn_conf * 100:.1f}%", fontsize=10, fontweight="bold"
            )
            axes[i, 1].axis("off")

            # Col 2: ResNet-18 Overlay
            axes[i, 2].imshow(overlay_r)
            axes[i, 2].set_title(
                f"ResNet-18 (2x2)\nConf: {res_conf * 100:.1f}%", fontsize=10, fontweight="bold"
            )
            axes[i, 2].axis("off")

    plt.suptitle(
        "Architecture Explainability Comparison: Custom CNN vs Adapted ResNet-18 across All 9 Classes",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    grid_cmp = compare_dir / "gradcam_comparison_9classes.png"
    fig.savefig(grid_cmp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_compare_paths.append(grid_cmp)

    # Clean up hooks
    cam_resnet.remove_hooks()
    cam_cnn.remove_hooks()

    logger.info(
        f"Phase 6B completed! Generated {len(saved_resnet_paths)} ResNet-18 figures and {len(saved_compare_paths)} comparative figures."
    )


if __name__ == "__main__":
    main()
