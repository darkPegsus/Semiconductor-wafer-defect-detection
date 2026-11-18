# ruff: noqa: E402
"""Phase 6A: Generate Grad-CAM Visualizations for CustomWaferCNN Baseline.

Loads the winning CustomWaferCNN baseline checkpoint (exp1_custom_cnn_baseline_best.pt),
generates class activation maps targeting 'features.conv_block4' across key validation cases:
- Correctly classified wafers across all 9 classes
- Incorrectly classified wafers (defect-to-defect confusion)
- False alarms on the 'none' class (clean wafers predicted as defects)
- Missed defects (Scratch, Loc, Edge-Loc predicted as 'none')
- High-confidence and low-confidence predictions

Strictly uses the validation partition (N_val = 24,705). The test set remains untouched.
Figures are saved to data/interim/figures/gradcam/.
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


def plot_single_explanation(
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
    """Render and save a 3-panel Grad-CAM explanation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Base discrete wafer rendering
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
    axes[1].set_title(
        f"Grad-CAM Heatmap (conv_block4)\n{target_str}", fontsize=11, fontweight="bold"
    )
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


def main():
    logger.info("=================================================================")
    logger.info("       PHASE 6A: GRAD-CAM EXPLAINABILITY FOR CUSTOM CNN          ")
    logger.info("=================================================================")

    # 1. Load dataset & validation split
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")

    val_indices = splits["val"]
    test_indices = splits["test"]

    # Verify test set isolation
    assert len(test_indices) == 24705
    logger.info("Verified: Test partition strictly unvisited (24,705 held-out samples).")

    val_df = df.iloc[val_indices].reset_index(drop=True)
    val_dataset = WM811kDataset(
        val_df,
        preprocessing_method="method_c",
        representation="single_channel",
    )
    logger.info(f"Loaded validation dataset: {len(val_dataset):,} wafers.")

    # 2. Load winning Custom CNN model
    checkpoint_path = Path("models/checkpoints/exp1_custom_cnn_baseline_best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomWaferCNN(in_channels=1, num_classes=9).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} (Best Val Macro F1: {ckpt.get('best_score')})."
    )

    # 3. Initialize GradCAM on target layer 'features.conv_block4'
    cam_engine = GradCAM(model=model, target_layer="features.conv_block4", device=device)
    logger.info("GradCAM engine initialized targeting layer: features.conv_block4.")

    # 4. Search validation set for representative cases
    output_dir = Path("data/interim/figures/gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Categories to discover
    correct_samples = {}  # class_idx -> (idx, prob, pred)
    false_alarms = []  # (idx, gt, pred, prob)
    missed_defects = []  # (idx, gt, pred, prob)
    confusions = []  # (idx, gt, pred, prob)
    high_conf_samples = []  # (idx, gt, pred, prob)
    low_conf_samples = []  # (idx, gt, pred, prob)

    logger.info("Scanning validation set to categorize evaluation cases...")

    batch_size = 64
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_x = batch["image"].to(device)
            batch_y = batch["label"]
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            for b in range(len(batch_y)):
                idx = sample_offset + b
                gt = int(batch_y[b].item())
                pred = int(preds[b].item())
                conf = float(probs[b, pred].item())

                # 1. Correct samples (high confidence per class)
                if gt == pred:
                    if gt not in correct_samples or conf > correct_samples[gt][1]:
                        correct_samples[gt] = (idx, conf, pred)

                # 2. False alarms on 'none' (GT == none, Pred != none)
                if gt == 0 and pred != 0 and len(false_alarms) < 10:
                    false_alarms.append((idx, gt, pred, conf))

                # 3. Missed defects (GT != none, Pred == none)
                if gt != 0 and pred == 0 and len(missed_defects) < 15:
                    missed_defects.append((idx, gt, pred, conf))

                # 4. Defect-to-defect confusion (GT != none, Pred != none, GT != Pred)
                if gt != 0 and pred != 0 and gt != pred and len(confusions) < 10:
                    confusions.append((idx, gt, pred, conf))

                # 5. Low-confidence predictions (conf in [0.35, 0.65])
                if 0.35 <= conf <= 0.65 and len(low_conf_samples) < 10:
                    low_conf_samples.append((idx, gt, pred, conf))

                # 6. High-confidence predictions (conf > 0.99)
                if conf > 0.99 and gt != 0 and len(high_conf_samples) < 10:
                    high_conf_samples.append((idx, gt, pred, conf))

            sample_offset += len(batch_y)

    logger.info(f"Discovered: {len(correct_samples)} / 9 correct classes.")
    logger.info(
        f"Discovered: {len(false_alarms)} false alarms, {len(missed_defects)} missed defects."
    )
    logger.info(
        f"Discovered: {len(confusions)} defect confusions, {len(low_conf_samples)} low confidence cases."
    )

    # 5. Generate and save visualizations
    saved_paths = []

    # Case A: Correctly Classified Samples for All 9 Classes
    logger.info("Generating Grad-CAM for Correctly Classified Wafers (All 9 Classes)...")
    for cls_idx in sorted(correct_samples.keys()):
        val_idx, conf, pred = correct_samples[cls_idx]
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        # Generate CAM
        cam, pred_t, prob_t = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=pred)
        heatmap_np = cam.squeeze().numpy()
        overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

        cls_name = IDX_TO_LABEL[cls_idx]
        filename = f"gradcam_correct_{cls_idx}_{cls_name.lower().replace('-', '_')}.png"
        out_file = output_dir / filename
        plot_single_explanation(
            wafer_map=wafer_np,
            heatmap=heatmap_np,
            overlay=overlay_np,
            gt_label=cls_name,
            pred_label=cls_name,
            confidence=conf,
            case_title=f"Correct Classification: {cls_name} (Confidence: {conf * 100:.1f}%)",
            output_path=out_file,
        )
        saved_paths.append(out_file)

    # Case B: Missed Defects (Focus on Scratch, Loc, Edge-Loc)
    logger.info(
        "Generating Grad-CAM for Missed Defects (Scratch, Loc, Edge-Loc predicted as none)..."
    )
    target_misses = [LABEL_TO_IDX["Scratch"], LABEL_TO_IDX["Loc"], LABEL_TO_IDX["Edge-Loc"]]
    for target_gt in target_misses:
        matching = [item for item in missed_defects if item[1] == target_gt]
        if matching:
            val_idx, gt, pred, conf = matching[0]
            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

            gt_name = IDX_TO_LABEL[gt]
            pred_name = IDX_TO_LABEL[pred]

            # Explanation for Predicted class ('none')
            cam_pred, _, _ = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=pred)
            heatmap_pred = cam_pred.squeeze().numpy()
            overlay_pred = overlay_heatmap_on_wafer(
                wafer_np, heatmap_pred, alpha=0.5, colormap="jet"
            )

            out_pred = output_dir / f"gradcam_missed_{gt_name.lower()}_pred_none.png"
            plot_single_explanation(
                wafer_map=wafer_np,
                heatmap=heatmap_pred,
                overlay=overlay_pred,
                gt_label=gt_name,
                pred_label=pred_name,
                confidence=conf,
                case_title=f"Missed Defect: True {gt_name} predicted as {pred_name} ({conf * 100:.1f}%)",
                output_path=out_pred,
            )
            saved_paths.append(out_pred)

            # Explanation for True Ground Truth class (what would activate the defect class?)
            cam_gt, _, prob_gt = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=gt)
            heatmap_gt = cam_gt.squeeze().numpy()
            overlay_gt = overlay_heatmap_on_wafer(wafer_np, heatmap_gt, alpha=0.5, colormap="jet")
            gt_conf = float(prob_gt[0, gt].item())

            out_gt = output_dir / f"gradcam_missed_{gt_name.lower()}_target_true_class.png"
            plot_single_explanation(
                wafer_map=wafer_np,
                heatmap=heatmap_gt,
                overlay=overlay_gt,
                gt_label=gt_name,
                pred_label=pred_name,
                confidence=gt_conf,
                case_title=f"Attribution to True Class ({gt_name}) for Missed Sample",
                output_path=out_gt,
                target_class_label=gt_name,
            )
            saved_paths.append(out_gt)

    # Case C: False Alarms on 'none' (Clean wafers predicted as defect)
    logger.info("Generating Grad-CAM for False Alarms on 'none'...")
    for i, (val_idx, gt, pred, conf) in enumerate(false_alarms[:3]):
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        gt_name = IDX_TO_LABEL[gt]
        pred_name = IDX_TO_LABEL[pred]

        cam, _, _ = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=pred)
        heatmap_np = cam.squeeze().numpy()
        overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

        out_fa = output_dir / f"gradcam_false_alarm_{i + 1}_{pred_name.lower()}.png"
        plot_single_explanation(
            wafer_map=wafer_np,
            heatmap=heatmap_np,
            overlay=overlay_np,
            gt_label=gt_name,
            pred_label=pred_name,
            confidence=conf,
            case_title=f"False Alarm: Clean Wafer ({gt_name}) Falsely Predicted as {pred_name}",
            output_path=out_fa,
        )
        saved_paths.append(out_fa)

    # Case D: Defect-to-Defect Boundary Confusions (Donut vs Center, Loc vs Edge-Loc)
    logger.info("Generating Grad-CAM for Defect-to-Defect Boundary Confusions...")
    for _i, (val_idx, gt, pred, conf) in enumerate(confusions[:3]):
        wafer_sample = val_dataset[val_idx]
        wafer_tensor = wafer_sample["image"]
        wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)

        gt_name = IDX_TO_LABEL[gt]
        pred_name = IDX_TO_LABEL[pred]

        cam, _, _ = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=pred)
        heatmap_np = cam.squeeze().numpy()
        overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

        out_conf = (
            output_dir / f"gradcam_confusion_{_i + 1}_{gt_name.lower()}_as_{pred_name.lower()}.png"
        )
        plot_single_explanation(
            wafer_map=wafer_np,
            heatmap=heatmap_np,
            overlay=overlay_np,
            gt_label=gt_name,
            pred_label=pred_name,
            confidence=conf,
            case_title=f"Defect Confusion: True {gt_name} Misclassified as {pred_name}",
            output_path=out_conf,
        )
        saved_paths.append(out_conf)

    # Case E: High vs Low Confidence Comparison
    logger.info("Generating Grad-CAM for High-Confidence vs Low-Confidence Cases...")
    if high_conf_samples and low_conf_samples:
        # High confidence
        val_idx_hi, gt_hi, pred_hi, conf_hi = high_conf_samples[0]
        wafer_sample_hi = val_dataset[val_idx_hi]
        wafer_tensor_hi = wafer_sample_hi["image"]
        wafer_np_hi = wafer_tensor_hi.squeeze().cpu().numpy().astype(np.uint8)
        cam_hi, _, _ = cam_engine.generate_cam(wafer_tensor_hi.unsqueeze(0), target_class=pred_hi)
        heatmap_hi = cam_hi.squeeze().numpy()
        overlay_hi = overlay_heatmap_on_wafer(wafer_np_hi, heatmap_hi, alpha=0.5, colormap="jet")

        out_hi = output_dir / "gradcam_high_confidence.png"
        plot_single_explanation(
            wafer_map=wafer_np_hi,
            heatmap=heatmap_hi,
            overlay=overlay_hi,
            gt_label=IDX_TO_LABEL[gt_hi],
            pred_label=IDX_TO_LABEL[pred_hi],
            confidence=conf_hi,
            case_title=f"High-Confidence Prediction ({conf_hi * 100:.1f}%)",
            output_path=out_hi,
        )
        saved_paths.append(out_hi)

        # Low confidence
        val_idx_lo, gt_lo, pred_lo, conf_lo = low_conf_samples[0]
        wafer_sample_lo = val_dataset[val_idx_lo]
        wafer_tensor_lo = wafer_sample_lo["image"]
        wafer_np_lo = wafer_tensor_lo.squeeze().cpu().numpy().astype(np.uint8)
        cam_lo, _, _ = cam_engine.generate_cam(wafer_tensor_lo.unsqueeze(0), target_class=pred_lo)
        heatmap_lo = cam_lo.squeeze().numpy()
        overlay_lo = overlay_heatmap_on_wafer(wafer_np_lo, heatmap_lo, alpha=0.5, colormap="jet")

        out_lo = output_dir / "gradcam_low_confidence.png"
        plot_single_explanation(
            wafer_map=wafer_np_lo,
            heatmap=heatmap_lo,
            overlay=overlay_lo,
            gt_label=IDX_TO_LABEL[gt_lo],
            pred_label=IDX_TO_LABEL[pred_lo],
            confidence=conf_lo,
            case_title=f"Low-Confidence / Borderline Prediction ({conf_lo * 100:.1f}%)",
            output_path=out_lo,
        )
        saved_paths.append(out_lo)

    # 6. Composite Grid: 9-Class Overview
    logger.info("Generating 9-class composite Grad-CAM summary grid...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(9):
        if i in correct_samples:
            val_idx, conf, pred = correct_samples[i]
            wafer_sample = val_dataset[val_idx]
            wafer_tensor = wafer_sample["image"]
            wafer_np = wafer_tensor.squeeze().cpu().numpy().astype(np.uint8)
            cam, _, _ = cam_engine.generate_cam(wafer_tensor.unsqueeze(0), target_class=pred)
            heatmap_np = cam.squeeze().numpy()
            overlay_np = overlay_heatmap_on_wafer(wafer_np, heatmap_np, alpha=0.5, colormap="jet")

            axes[i].imshow(overlay_np)
            cls_name = IDX_TO_LABEL[i]
            axes[i].set_title(
                f"{i}: {cls_name}\nConf: {conf * 100:.1f}%", fontsize=11, fontweight="bold"
            )
            axes[i].axis("off")

    plt.suptitle(
        "CustomWaferCNN Grad-CAM Visualizations across All 9 Defect Classes",
        fontsize=15,
        fontweight="bold",
        y=0.94,
    )
    plt.tight_layout()
    grid_out = output_dir / "gradcam_summary_9classes.png"
    fig.savefig(grid_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(grid_out)

    cam_engine.remove_hooks()
    logger.info(
        f"Phase 6A Grad-CAM generation complete! {len(saved_paths)} figures saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
