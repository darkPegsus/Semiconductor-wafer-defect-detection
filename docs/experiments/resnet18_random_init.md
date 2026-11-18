# Phase 5B: ResNet-18 Random Initialization Training (ResNet18-B) — Experimental Report

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Model Family:** Residual Neural Networks (`AdaptedResNet18`)  
**Experiment Identifier:** Phase 5B / ResNet18-B  
**Evaluation Partition:** Validation Set ($N_{\text{val}} = 24,705$ wafers, 1,537 lots, zero lot overlap)  
**Held-Out Benchmark:** Test Set ($N_{\text{test}} = 24,705$ wafers, **strictly unvisited**)  
**Benchmark Date:** November 2026  
**Status:** Completed & Synchronized (Best Checkpoint: Epoch 10, Early Stopping Triggered at Epoch 17)  

---

## 1. Executive Summary

Phase 5B benchmarked the same `AdaptedResNet18` architecture used in Phase 5A, but trained entirely from random initialization (Kaiming Normal, without ImageNet pretrained weights), to isolate the precise impact of transfer learning versus training from scratch on semiconductor wafer defect classification.

### Primary Experimental Findings:
- **Best Validation Macro F1:** **0.8544** (achieved at **Epoch 10**)
- **Validation Accuracy:** **97.05%**
- **Macro Precision:** **0.8507**
- **Macro Recall:** **0.8609**
- **Validation Loss:** **0.0958** (Train Loss at best epoch: **0.0444**)
- **Early Stopping:** Triggered at Epoch 17 after 7 consecutive epochs without exceeding minimum delta improvement ($\Delta \ge 0.001$).
- **Comparison to Phase 5A (Pretrained ResNet-18):** Random initialization achieved **$-0.0067$ lower Macro F1** ($0.8544$ vs. $0.8611$), demonstrating that ImageNet transfer learning provided a modest precision benefit, but did not overcome the structural inductive bias limitation of the ResNet stem.
- **Comparison to Custom CNN Baseline (Exp 1):** Underperformed `CustomWaferCNN` by **$-0.0317$ Macro F1** ($0.8544$ vs. $0.8861$), confirming that a wafer-tailored architecture preserving spatial die connectivity outperforms large generic backbones on discrete wafer maps.

```
                              Validation Macro F1 Benchmark
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Exp 1: Custom CNN Baseline (Standard CE)     ██████████████████████████████ 0.8861 (WIN)│
  │ Exp 3: Custom CNN Focal Loss                 █████████████████████████████▍ 0.8846      │
  │ Exp 2: Custom CNN Weighted CE                ████████████████████████████▎  0.8831      │
  │ Exp 4: Custom CNN Weighted Sampling          ████████████████████████████▏  0.8828      │
  │ ResNet18-A: Pretrained ResNet-18 (Phase 5A)  ███████████████████████████   0.8611      │
  │ ResNet18-B: Random Init ResNet-18 (Phase 5B) ██████████████████████████   0.8544      │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experimental Controls & Training Configuration

To maintain strict experimental parity with Phase 5A, all data pipeline and architectural parameters were held constant:

| Experimental Variable | Phase 5A (Pretrained) | Phase 5B (Random Init) | Rationale & Control |
|:---|:---|:---|:---|
| **Architecture** | `AdaptedResNet18` | `AdaptedResNet18` | Identical residual blocks, stem, and classifier |
| **Weight Initialization** | `ResNet18_Weights.DEFAULT` | **Kaiming Normal (`pretrained=False`)** | Standard PyTorch random initialization |
| **Total Parameters** | 11,174,857 | 11,174,857 | Exact parameter match |
| **Trainable Parameters** | 7,753 (Warmup) $\to$ 11.2M | **11,174,857 (100% from Epoch 1)** | No backbone freezing needed from scratch |
| **Input Representation** | Single-channel numerical (`[B, 1, 64, 64]`) | Single-channel numerical (`[B, 1, 64, 64]`) | Identical discrete $\{0, 1, 2\}$ representation |
| **Preprocessing** | Method C (Letterbox Padding) | Method C (Letterbox Padding) | Canonical $64 \times 64$ aspect-ratio preserving |
| **Loss Function** | Standard Cross-Entropy | Standard Cross-Entropy | Unweighted multi-class negative log-likelihood |
| **Optimizer** | AdamW (`weight_decay=1e-2`) | AdamW (`weight_decay=1e-2`) | Decoupled weight decay regularization |
| **Initial Learning Rate** | Warmup $10^{-4} \to$ Backbone $10^{-5}$ / Head $10^{-4}$ | **$1.0 \times 10^{-3}$ (uniform across all layers)** | Standard rate for training from scratch |
| **Scheduler** | CosineAnnealingLR | CosineAnnealingLR ($T_{\max}=30, \eta_{\min}=10^{-6}$) | Continuous cosine decay |
| **Batch Size** | 32 (Train) / 64 (Val) | 32 (Train) / 64 (Val) | Identical memory and gradient batching |
| **Precision** | Mixed Precision (`torch.amp`) | Mixed Precision (`torch.amp`) | FP16 Autocast with `GradScaler` |
| **Early Stopping** | Validation Macro F1 ($p=7$) | Validation Macro F1 ($p=7$) | Monitored with identical patience |
| **Dataset Partitions** | Group-Stratified 70/15/15 | Group-Stratified 70/15/15 | Zero lot overlap ($0\%$ leakage) |

---

## 3. Global Benchmark Matrix: ResNet18-B vs. ResNet18-A vs. Custom CNN

Validation partition ($N_{\text{val}} = 24,705$ wafers, 1,537 lots, zero lot leakage):

| Metric | Custom CNN Exp 1 (Baseline CE) | ResNet18-A (Pretrained ResNet) | ResNet18-B (Random Init ResNet) | Comparison: 5B vs 5A | Comparison: 5B vs Exp 1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Architecture** | `CustomWaferCNN` | `AdaptedResNet18` | `AdaptedResNet18` | Identical | $21.5\times$ larger |
| **Parameters** | **520,681** | 11,174,857 | 11,174,857 | Parity | $+10,654,176$ |
| **Initialization** | Random Init | ImageNet-1K Pretrained | **Random Kaiming Normal** | Contrast | Parity (scratch) |
| **Validation Macro F1** | **0.8861** | 0.8611 | **0.8544** | **$-0.0067$** | **$-0.0317$** |
| **Validation Accuracy** | **0.9765** | 0.9732 | **0.9705** | **$-0.0027$** | **$-0.0060$** |
| **Macro Precision** | **0.8943** | 0.8846 | **0.8507** | **$-0.0339$** | **$-0.0436$** |
| **Macro Recall** | **0.8813** | 0.8436 | **0.8609** | **$+0.0173$** | **$-0.0204$** |
| **Best Epoch** | Epoch 24 | Epoch 23 | **Epoch 10** | Earlier peak | Earlier peak |
| **Epochs Trained** | 30 / 30 | 30 / 30 | **17 / 30 (Early Stopped)** | $-13$ epochs | $-13$ epochs |
| **Active Duration** | **40.5 min** | 94.7 min | **82.8 min** | $-11.9$ min | $+42.3$ min |
| **Avg Epoch Duration** | **~81.0 s** | 189.5 s | 292.2 s | $+102.7$ s | $+211.2$ s |
| **Peak Active VRAM** | **125.2 MB** | 233.5 MB | 232.3 MB | $-1.2$ MB | $+107.1$ MB |
| **Clean False Alarms** | 209 (0.99%) | **126 (0.60%)** | 282 (1.34%) | $+156$ wafers | $+73$ wafers |

---

## 4. Complete Per-Class Performance Breakdown

Validation partition ($N_{\text{val}} = 24,705$ wafers). Metrics formatted as **Precision / Recall / F1-Score**:

| Class | Support | Custom CNN Exp 1 (Baseline) | ResNet18-A (Pretrained ResNet) | ResNet18-B (Random Init ResNet) | Top Strategy | Impact of Pretraining (5A vs 5B) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`none`** | 21,061 | 0.9895 / 0.9901 / **0.9898** | 0.9833 / **0.9940** / 0.9886 | **0.9886** / 0.9866 / 0.9876 | **Custom CNN Exp 1** | Pretrained has higher recall (+0.74%) |
| **`Center`** | 613 | 0.9220 / 0.9250 / **0.9235** | **0.9269** / 0.8891 / 0.9076 | 0.8739 / **0.9494** / 0.9101 | **Custom CNN Exp 1** | Random init has higher recall (+6.03%) |
| **`Donut`** | 79 | **0.8734** / **0.8734** / **0.8734** | 0.7791 / 0.8481 / 0.8121 | 0.8592 / 0.7722 / 0.8133 | **Custom CNN Exp 1** | Comparable F1 (0.8121 vs 0.8133) |
| **`Edge-Loc`**| 741 | 0.8198 / **0.8596** / **0.8393** | **0.8511** / 0.7557 / 0.8006 | 0.7984 / 0.7962 / 0.7973 | **Custom CNN Exp 1** | Pretrained has higher precision (+5.27%) |
| **`Edge-Ring`**| 1,383 | **0.9832** / **0.9747** / **0.9789** | 0.9789 / 0.9711 / 0.9750 | 0.9732 / 0.9711 / 0.9721 | **Custom CNN Exp 1** | Comparable F1 across all backbones |
| **`Loc`** | 513 | 0.7883 / **0.7914** / **0.7899** | **0.8072** / 0.7018 / 0.7508 | 0.7556 / 0.7173 / 0.7360 | **Custom CNN Exp 1** | Pretrained has higher precision (+5.16%) |
| **`Random`** | 124 | 0.9153 / 0.8710 / 0.8926 | **0.9386** / 0.8629 / 0.8992 | 0.9316 / **0.8790** / **0.9046** | **ResNet18-B (0.9046)** | Random init achieves highest F1 |
| **`Scratch`** | 170 | **0.8872** / 0.6941 / **0.7789** | 0.8268 / 0.6176 / 0.7071 | 0.6059 / **0.7235** / 0.6595 | **Custom CNN Exp 1** | Random init has higher recall, lower prec |
| **`Near-full`**| 21 | 0.8696 / **0.9524** / 0.9091 | 0.8696 / **0.9524** / 0.9091 | 0.8696 / **0.9524** / 0.9091 | **Tied across all** | Exact parity (20/21 detected) |

---

## 5. Confusion Matrix Analysis & Defect Escape Dynamics

Validation partition ($N_{\text{val}} = 24,705$ wafers: 21,061 true `none`, 3,644 true defective):

```
               Confusion Matrix Summary (ResNet18-B, Epoch 10)
  Predicted ->  none  Center  Donut  Edge-Loc  Edge-Ring  Loc  Random  Scratch  Near-full
  True Class
  none         20779      52      0        92         17   65       1       55          0
  Center          29     582      1         0          0    1       0        0          0
  Donut            0      12     61         0          0    5       1        0          0
  Edge-Loc        81       2      0       590         19   35       4        8          2
  Edge-Ring       15       0      0        23       1343    1       1        0          0
  Loc             79      17      7        24          0  368       1       17          0
  Random           3       1      0         6          1    3     109        0          1
  Scratch         33       0      1         4          0    9       0      123          0
  Near-full        0       0      1         0          0    0       0        0         20
```

### Detailed Observations:
1. **Clean Wafer False Alarms (1.34%):**
   - ResNet18-B generated **282 false alarms** out of 21,061 clean wafers.
   - This is higher than ResNet18-A (126 false alarms, 0.60%) and Custom CNN Exp 1 (209 false alarms, 0.99%).
   - The majority of false alarms occurred on: `Edge-Loc` (92), `Loc` (65), `Scratch` (55), and `Center` (52).
2. **Defect Escape to Clean (`Class` $\to$ `none`):**
   - **`Scratch` Escapes:** ResNet18-B missed only **33 scratches** (72.35% recall, 123 detected), outperforming ResNet18-A which missed 52 scratches (61.76% recall). However, ResNet18-B predicted 55 clean wafers as scratches, driving its precision down to 60.59%.
   - **`Loc` Escapes:** 79 `Loc` wafers escaped to clean ($15.40\%$), compared to 97 in ResNet18-A ($18.91\%$) and 70 in Custom CNN Exp 1 ($13.65\%$).
   - **`Edge-Loc` Escapes:** 81 `Edge-Loc` wafers escaped to clean ($10.93\%$), compared to 135 in ResNet18-A ($18.22\%$).
3. **Defect-to-Defect Boundary Confusions:**
   - Concentric confusion: 12 `Donut` wafers were misclassified as `Center` (15.19% confusion) due to shared central radial symmetry.
   - Spatial proximity: 35 `Edge-Loc` wafers were classified as `Loc`, and 24 `Loc` wafers were classified as `Edge-Loc`.

---

## 6. Overfitting Observations & Training Dynamics

### 6.1 Training Loss vs. Validation Divergence
- **Rapid Memorization:** ResNet18-B's training loss plunged from **0.3745** (Epoch 1) to **0.0444** (Epoch 10) and **0.0075** (Epoch 17).
- **Validation Loss Minimum:** Validation loss reached its minimum of **0.0889 at Epoch 6** and **0.0958 at Epoch 10**, before diverging to **0.1956 at Epoch 17**.
- **Early Stopping:** The network plateaued at Epoch 10 (Macro F1 = 0.8544). Subsequent epochs (Epoch 11–17) failed to improve by more than the minimum delta ($\Delta \ge 0.001$), triggering early termination at Epoch 17.

### 6.2 Pretrained vs. Random Initialization Insights
1. **Pretrained Features Provide Regularization and Precision:**
   - ImageNet pretraining acted as a strong prior that restrained early overparameterization. ResNet18-A attained **0.8846 Macro Precision** and only 126 clean false alarms, whereas ResNet18-B attained **0.8507 Macro Precision** with 282 clean false alarms.
2. **Random Initialization Unlocks Higher Defect Sensitivity (Recall):**
   - Because the filters were not pre-conditioned on smooth natural photography, ResNet18-B learned sharper high-frequency filters from scratch, increasing recall on `Scratch` (+10.59% vs. 5A) and `Center` (+6.03% vs. 5A), but at the expense of false positives.
3. **Architectural Inductive Bias is the Primary Bottleneck:**
   - The performance gap between ResNet-18 pretrained and random init is relatively minor ($\Delta \text{F1} = 0.0067$).
   - However, the performance gap between both ResNet-18 variants and `CustomWaferCNN` is substantial ($\Delta \text{F1} = 0.0250$ to $0.0317$).
   - **Conclusion:** Pretraining cannot compensate for a convolutional stem that prematurely downsamples $64 \times 64$ die grids to $16 \times 16$ before deep feature processing.

---

## 7. Hardware & Training Profiling (NVIDIA RTX 2050 4GB)

- **Total Active Training Duration:** 82.80 minutes ($4,967.7\text{ s}$) across 17 epochs.
- **Average Epoch Duration:** **292.2 seconds / epoch**
  - Slightly higher per-epoch compute time compared to Phase 5A because 100% of the 11.17M parameters required backward gradient computation throughout all epochs (unlike Phase 5A's 3-epoch warmup where 99.9% of parameters were frozen).
- **VRAM Memory Architecture:**
  - Peak PyTorch Active Allocated VRAM: **232.3 MB**
  - Peak PyTorch Reserved VRAM: **272.0 MB**
  - Process VRAM (`nvidia-smi`): Monitored within 3,035 MB driver allocation ceiling during peak data transfers.
  - Average GPU Utilization: **59.3%**

---

## 8. Local Artifacts & Reproducibility Paths

| Artifact Type | Path | Description |
|:---|:---|:---|
| **Best Model Checkpoint** | [`models/checkpoints/resnet18_random_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/resnet18_random_best.pt) | Best checkpoint (Epoch 10, Macro F1: 0.8544, 44.7 MB) |
| **Training Curves Plot** | [`data/interim/figures/resnet18_random_training_curves.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_random_training_curves.png) | Epoch-wise train/val loss, macro precision, recall, and F1 |
| **Confusion Matrix Plot** | [`data/interim/figures/resnet18_random_confusion_matrix.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_random_confusion_matrix.png) | Raw and normalized validation confusion matrix |
| **Complete Metrics JSON** | [`data/interim/figures/resnet18_random_metrics.json`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_random_metrics.json) | Full numerical history, per-class metrics, timing, and VRAM |
| **Training Runner Script** | [`scripts/train_resnet18_random.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/train_resnet18_random.py) | Reproducible training script with Kaiming Normal initialization |
