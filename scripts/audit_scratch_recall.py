# ruff: noqa: E402
"""Audit Scratch Recall and Complete Validation Metrics for Custom CNN and ResNet18."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataset import IDX_TO_LABEL, WM811kDataset, load_labeled_dataset
from src.data.group_split import load_split_indices
from src.models.custom_cnn import CustomWaferCNN
from src.models.resnet18 import AdaptedResNet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")
    val_indices = splits["val"]
    test_indices = splits["test"]
    assert len(test_indices) == 24705, f"Test leakage! {len(test_indices)}"
    print(
        f"Validation partition size: {len(val_indices):,} (Test strictly isolated: {len(test_indices):,})"
    )

    val_df = df.iloc[val_indices].reset_index(drop=True)
    val_dataset = WM811kDataset(
        val_df, preprocessing_method="method_c", representation="single_channel"
    )
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    # 1. Custom CNN
    cnn_path = Path("models/checkpoints/exp1_custom_cnn_baseline_best.pt")
    cnn_ckpt = torch.load(cnn_path, map_location=device)
    cnn_model = CustomWaferCNN(in_channels=1, num_classes=9).to(device)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn_model.eval()

    # 2. ResNet18
    resnet_path = Path("models/checkpoints/resnet18_pretrained_best.pt")
    resnet_ckpt = torch.load(resnet_path, map_location=device)
    resnet_model = AdaptedResNet18(in_channels=1, num_classes=9, pretrained=False).to(device)
    resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
    resnet_model.eval()

    y_true = []
    y_pred_cnn = []
    y_pred_resnet = []

    with torch.no_grad():
        for batch in loader:
            bx = batch["image"].to(device)
            by = batch["label"].numpy()

            logits_c = cnn_model(bx)
            p_c = torch.argmax(logits_c, dim=1).cpu().numpy()

            logits_r = resnet_model(bx)
            p_r = torch.argmax(logits_r, dim=1).cpu().numpy()

            y_true.extend(by)
            y_pred_cnn.extend(p_c)
            y_pred_resnet.extend(p_r)

    y_true = np.array(y_true)
    y_pred_cnn = np.array(y_pred_cnn)
    y_pred_resnet = np.array(y_pred_resnet)

    print("\n" + "=" * 90)
    print("INDEPENDENT VALIDATION AUDIT RESULTS (REPRODUCIBILITY CHECK)")
    print("=" * 90)

    for name, preds in [
        ("CustomWaferCNN (Exp 1 Baseline)", y_pred_cnn),
        ("AdaptedResNet18 (Pretrained)", y_pred_resnet),
    ]:
        print(f"\n--- {name} ---")
        cm = confusion_matrix(y_true, preds, labels=list(range(9)))
        rep = classification_report(
            y_true,
            preds,
            labels=list(range(9)),
            target_names=[IDX_TO_LABEL[i] for i in range(9)],
            digits=4,
            output_dict=True,
        )

        acc = rep["accuracy"]
        mf1 = rep["macro avg"]["f1-score"]
        mrec = rep["macro avg"]["recall"]
        mprec = rep["macro avg"]["precision"]

        print(f"Overall Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Macro F1:        {mf1:.4f}")
        print(f"Macro Recall:    {mrec:.4f} ({mrec * 100:.2f}%)")
        print(f"Macro Precision: {mprec:.4f} ({mprec * 100:.2f}%)")

        print("\nPer-class Metrics:")
        header = f"{'Class':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Support':<7} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Escapes to none':<18}"
        print(header)
        print("-" * len(header))
        for i in range(9):
            cname = IDX_TO_LABEL[i]
            tp = int(cm[i, i])
            fn = int(np.sum(cm[i, :]) - tp)
            fp = int(np.sum(cm[:, i]) - tp)
            supp = int(rep[cname]["support"])
            prec = rep[cname]["precision"]
            rec = rep[cname]["recall"]
            f1 = rep[cname]["f1-score"]
            escapes = int(cm[i, 0]) if i != 0 else 0
            esc_str = f"{escapes} ({escapes / supp * 100:.2f}%)" if i != 0 else "N/A (clean)"
            print(
                f"{cname:<10} | {tp:<5} | {fp:<5} | {fn:<5} | {supp:<7} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {esc_str:<18}"
            )

        # Confusion with scratch
        scratch_idx = 7
        print(f"\nBreakdown of true 'Scratch' (Support = {rep['Scratch']['support']}):")
        for j in range(9):
            count = int(cm[scratch_idx, j])
            if count > 0:
                print(f"  Predicted as {IDX_TO_LABEL[j]:<10}: {count} ({count / 170 * 100:.2f}%)")


if __name__ == "__main__":
    main()
