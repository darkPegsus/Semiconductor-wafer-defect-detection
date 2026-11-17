# Phase 5A: ResNet-18 ImageNet Transfer Learning (ResNet18-A) — Experimental Report

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Model Family:** Residual Neural Networks (`AdaptedResNet18`)  
**Experiment Identifier:** Phase 5A / ResNet18-A  
**Evaluation Partition:** Validation Set ($N_{\text{val}} = 24,705$ wafers, 1,537 lots, zero lot overlap)  
**Held-Out Benchmark:** Test Set ($N_{\text{test}} = 24,705$ wafers, **strictly unvisited**)  
**Benchmark Date:** November 2026  
**Status:** Completed & Synchronized (Best Checkpoint: Epoch 23, Early Stopping Triggered at Epoch 30)  

---

## 1. Executive Summary

Phase 5A evaluated whether leveraging general visual feature representations learned from natural images (ImageNet-1K) via transfer learning improves spatial defect classification over a domain-tailored CNN trained from scratch on semiconductor wafer maps.

### Primary Experimental Findings:
- **Best Validation Macro F1:** **0.8611** (achieved at **Epoch 23**)
- **Validation Accuracy:** **97.32%**
- **Macro Precision:** **0.8846**
- **Macro Recall:** **0.8436**
- **Validation Loss:** **0.3275** (Train Loss at best epoch: **0.0018**)
- **Training Progression:** 30 / 30 Epochs trained; early stopping patience ($p=7$) exhausted following Epoch 23 peak.
- **Comparison to Custom CNN Baseline:** Underperformed Custom CNN Experiment 1 (Baseline Standard CE) by **$-0.0250$ Macro F1** ($0.8611$ vs. $0.8861$).

```
                                Validation Macro F1 Benchmark
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ Exp 1: Custom CNN Baseline (Standard CE)     ██████████████████████████████ 0.8861 (WIN)│
  │ Exp 3: Custom CNN Focal Loss                 █████████████████████████████▍ 0.8846      │
  │ Exp 2: Custom CNN Weighted CE                ████████████████████████████▎  0.8831      │
  │ Exp 4: Custom CNN Weighted Sampling          ████████████████████████████▏  0.8828      │
  │ ResNet18-A: Pretrained ResNet-18 (Phase 5A)  ██───────────────────────────  0.8611      │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Architecture & Adaptation Strategy

The network architecture is implemented in [`src/models/resnet18.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/src/models/resnet18.py) as `AdaptedResNet18`, wrapping TorchVision's standard ResNet-18 topology.

### 2.1 Initialization
- Initialized with official PyTorch weights: `ResNet18_Weights.DEFAULT` (ImageNet-1K, Top-1 Acc: 69.758%).
- Weights cached locally at `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (44.7 MB).

### 2.2 Single-Channel Input Adaptation (Weight Migration)
Standard ResNet-18 accepts 3-channel RGB inputs ($[B, 3, H, W]$) via its initial $7 \times 7$, stride-2 convolution `conv1` ($64 \times 3 \times 7 \times 7$ weights). For single-channel numerical wafer maps ($[B, 1, 64, 64]$), input channels were adapted by taking the cross-channel arithmetic mean of the pretrained weights:
$$W_{\text{adapted}}[:, 0:1, :, :] = \frac{1}{3} \sum_{c=0}^2 W_{\text{pretrained}}[:, c:c+1, :, :]$$
This mathematical migration preserves the low-level spatial filter responses (edges, circles, textures, and orthogonal gradients) without random re-initialization.

### 2.3 Classification Head
The original 1000-way linear classifier was replaced with:
$$\text{Dropout}(p=0.40) \longrightarrow \text{Linear}(512, 9)$$
producing raw, unnormalized logits for the 9 canonical wafer defect classes.

### 2.4 Exact Parameter Counts

| Stage / Component | Total Parameters | Trainable Parameters | Non-Trainable (Frozen) |
|:---|:---:|:---:|:---:|
| **Full AdaptedResNet18** | **11,174,857** | — | — |
| **Stage 1 (Warmup: Epochs 1–3)** | 11,174,857 | **7,753** (0.07%) | **11,167,104** (99.93%) |
| • `conv1` (adapted layer) | 3,136 | 3,136 | 0 |
| • `fc` (classification head) | 4,617 (512 $\times$ 9 + 9) | 4,617 | 0 |
| • Backbone (`bn1`, `layer1`–`layer4`) | 11,167,104 | 0 | 11,167,104 |
| **Stage 2 (Fine-Tuning: Epochs 4–30)** | **11,174,857** | **11,174,857** (100.0%) | **0** (0.0%) |

---

## 3. Two-Stage Training Protocol & Differential Learning Rates

To prevent catastrophic forgetting of pretrained feature extractors during initial gradient steps, a two-stage training scheme was executed:

### Stage 1: Classifier Warmup (Epochs 1–3)
- **Status:** Backbone frozen (`model.freeze_backbone()`). Only `conv1` and `fc` receive gradient updates.
- **Learning Rate:** $1.0 \times 10^{-4}$ on trainable parameters.
- **Scheduler:** 3-epoch warmup `CosineAnnealingLR` decaying to $1.0 \times 10^{-6}$.
- **Performance:** Validation Macro F1 rose from **0.5603** (Epoch 1) $\to$ **0.5999** (Epoch 2) $\to$ **0.6072** (Epoch 3).

### Stage 2: Full Differential Fine-Tuning (Epochs 4–30)
- **Status:** Backbone fully unfrozen at start of Epoch 4 (`model.unfreeze_backbone()`).
- **Differential Parameter Groups:**
  1. **Backbone Group (`conv1`, `bn1`, `layer1`–`layer4`):** Learning rate $= \mathbf{1.0 \times 10^{-5}}$ (10$\times$ lower to preserve delicate hierarchical spatial filters).
  2. **Classifier Head Group (`fc`):** Learning rate $= \mathbf{1.0 \times 10^{-4}}$.
- **Scheduler:** 27-epoch `CosineAnnealingLR` decaying to minimum $\eta_{\min} = 1.0 \times 10^{-6}$.
- **Immediate Impact:** Validation Macro F1 jumped instantly from **0.6072 $\to$ 0.7546 (+0.1474)** on Epoch 4.

---

## 4. Dataset Partitions & Hygiene Verification

All training and evaluation used the centralized partition defined in `data/interim/split_indices.json`:

| Partition | Wafer Count | Percentage | Manufacturing Lots | Lot Overlap Verification |
|:---|:---:|:---:|:---:|:---:|
| **Train Set** | 123,540 | 71.43% | 7,688 lots | $\text{Lots}_{\text{train}} \cap \text{Lots}_{\text{val}} = \emptyset$ |
| **Validation Set** | 24,705 | 14.28% | 1,537 lots | $\text{Lots}_{\text{train}} \cap \text{Lots}_{\text{test}} = \emptyset$ |
| **Test Set** | 24,705 | 14.28% | 1,537 lots | $\text{Lots}_{\text{val}} \cap \text{Lots}_{\text{test}} = \emptyset$ |

- **Verification:** Group-stratified isolation confirms **0% lot leakage** ($0$ common lots across all splits).
- **Test Set Protection:** The test set ($N_{\text{test}} = 24,705$) was **strictly excluded** from data loaders, model selection, loss calculation, and evaluation.

---

## 5. Global Validation Performance Comparison

Validation set ($N_{\text{val}} = 24,705$ wafers). ResNet18-A vs. All 4 Custom CNN Experiments:

| Metric | Exp 1: Baseline (Standard CE) | Exp 2: Weighted CE (Effective-Num) | Exp 3: Focal Loss ($\gamma=2.0$) | Exp 4: Weighted Sampling | ResNet18-A (Pretrained ResNet-18) | Top Performer |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Architecture** | `CustomWaferCNN` | `CustomWaferCNN` | `CustomWaferCNN` | `CustomWaferCNN` | `AdaptedResNet18` | — |
| **Parameters** | **520,681** | **520,681** | **520,681** | **520,681** | 11,174,857 | **Custom CNN ($21.5\times$ smaller)** |
| **Weights** | Random Init | Random Init | Random Init | Random Init | ImageNet-1K Pretrained | — |
| **Loss** | Standard CE | Weighted CE | Multi-Class Focal | Standard CE | Standard CE | — |
| **Sampler** | Uniform | Uniform | Uniform | Weighted Random | Uniform | — |
| **Validation Macro F1** | **0.8861** | 0.8831 | 0.8846 | 0.8828 | 0.8611 | **Exp 1 (Custom CNN Baseline)** |
| **Validation Accuracy** | 0.9765 | 0.9759 | **0.9766** | 0.9754 | 0.9732 | **Exp 3 (Focal Loss)** |
| **Macro Precision** | 0.8943 | 0.8783 | **0.9098** | 0.9036 | 0.8846 | **Exp 3 (Focal Loss)** |
| **Macro Recall** | 0.8813 | **0.8901** | 0.8639 | 0.8661 | 0.8436 | **Exp 2 (Weighted CE)** |
| **Best Epoch** | Epoch 24 | Epoch 16 | Epoch 24 | Epoch 25 | Epoch 23 | — |
| **Epochs Trained** | 30 / 30 | 23 / 30 (Stopped) | 30 / 30 | 30 / 30 | 30 / 30 (Stopped) | — |
| **Active Duration** | 40.5 min | **29.6 min** | 51.8 min | 50.9 min | 94.7 min | **Exp 2 (Fastest)** |
| **Avg Epoch Duration** | ~81.0 s | **77.3 s** | 103.6 s | 101.8 s | 189.5 s | **Exp 2** |
| **Peak Active VRAM** | **125.2 MB** | **125.2 MB** | 125.7 MB | 125.7 MB | 233.5 MB | **Custom CNN** |
| **Process VRAM (smi)** | 375 MB | 375 MB | 406 MB | 392 MB | 386 MB | Under 410 MB |

---

## 6. Complete Per-Class Metrics Matrix

Validation partition ($N_{\text{val}} = 24,705$ wafers). Metrics reported as **Precision / Recall / F1-Score**:

| Class | Support | Exp 1: Custom CNN (Baseline) | Exp 2: Custom CNN (Weighted CE) | Exp 3: Custom CNN (Focal Loss) | Exp 4: Custom CNN (Weighted Sampler) | ResNet18-A (Pretrained ResNet) | Top Strategy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`none`** | 21,061 | 0.9895 / 0.9901 / 0.9898 | 0.9902 / 0.9895 / 0.9898 | 0.9854 / **0.9943** / **0.9899** | 0.9854 / 0.9925 / 0.9890 | 0.9833 / 0.9940 / 0.9886 | **Exp 3 (Focal)** |
| **`Center`** | 613 | 0.9220 / 0.9250 / 0.9235 | 0.9269 / **0.9315** / **0.9292** | 0.9293 / 0.9217 / 0.9255 | **0.9318** / 0.9135 / 0.9226 | 0.9269 / 0.8891 / 0.9076 | **Exp 2 (Weighted CE)** |
| **`Donut`** | 79 | **0.8734** / 0.8734 / 0.8734 | 0.8085 / **0.9620** / **0.8786** | 0.8608 / 0.8608 / 0.8608 | 0.8072 / 0.8481 / 0.8272 | 0.7791 / 0.8481 / 0.8121 | **Exp 2 (Weighted CE)** |
| **`Edge-Loc`**| 741 | 0.8198 / **0.8596** / **0.8393** | 0.8049 / 0.8408 / 0.8224 | 0.8419 / 0.8192 / 0.8304 | 0.8347 / 0.8313 / 0.8330 | **0.8511** / 0.7557 / 0.8006 | **Exp 1 (Custom CNN)** |
| **`Edge-Ring`**| 1,383 | 0.9832 / 0.9747 / **0.9789** | 0.9832 / 0.9740 / 0.9786 | **0.9860** / 0.9696 / 0.9778 | 0.9839 / 0.9732 / 0.9786 | 0.9789 / 0.9711 / 0.9750 | **Exp 1 (Custom CNN)** |
| **`Loc`** | 513 | 0.7883 / **0.7914** / **0.7899** | 0.7901 / 0.7778 / 0.7839 | **0.8870** / 0.7037 / 0.7848 | 0.8476 / 0.6940 / 0.7631 | 0.8072 / 0.7018 / 0.7508 | **Exp 1 (Custom CNN)** |
| **`Random`** | 124 | 0.9153 / 0.8710 / 0.8926 | 0.8740 / 0.8952 / 0.8845 | 0.8926 / 0.8710 / 0.8816 | 0.8976 / **0.9194** / **0.9084** | **0.9386** / 0.8629 / 0.8992 | **Exp 4 (Weighted Sampler)** |
| **`Scratch`** | 170 | **0.8872** / 0.6941 / **0.7789** | 0.8224 / **0.7353** / 0.7764 | 0.8529 / 0.6824 / 0.7582 | 0.8444 / 0.6706 / 0.7475 | 0.8268 / 0.6176 / 0.7071 | **Exp 1 (F1) / Exp 2 (Recall)** |
| **`Near-full`**| 21 | 0.8696 / **0.9524** / 0.9091 | 0.9048 / 0.9048 / 0.9048 | 0.9524 / **0.9524** / 0.9524 | **1.0000** / **0.9524** / **0.9756** | 0.8696 / **0.9524** / 0.9091 | **Exp 4 (Weighted Sampler)** |

---

## 7. Confusion Matrix Findings & Defect Escape Dynamics

Validation partition: 21,061 true `none` wafers, 3,644 true defective wafers:

```
               Confusion Matrix Summary (ResNet18-A, Epoch 23)
  Predicted ->  none  Center  Donut  Edge-Loc  Edge-Ring  Loc  Random  Scratch  Near-full
  True Class
  none         20935      24      2        47          7   34       1       11          0
  Center          47     545      7         0          0   14       0        0          0
  Donut            0       7     67         0          0    5       0        0          0
  Edge-Loc       135       1      0       560         19   20       3        3          0
  Edge-Ring       23       0      2        14       1343    0       1        0          0
  Loc             97      10      6        28          2  360       2        8          0
  Random           2       1      2         6          1    2     107        0          3
  Scratch         52       0      0         2          0   11       0      105          0
  Near-full        0       0      0         1          0    0       0        0         20
```

### Key Observations:
1. **Clean Wafer False Alarm Rate (0.60%):**
   - ResNet18-A falsely flagged only **126 out of 21,061** clean wafers as defective (99.40% recall on `none`).
   - This outperforms Custom CNN Exp 1 (209 false alarms, 0.99%) and Exp 2 (222 false alarms, 1.05%), and approaches Focal Loss Exp 3 (119 false alarms, 0.57%).
2. **Defect Escape to Clean (`Class` $\to$ `none`):**
   - **`Scratch` Escapes:** 52 out of 170 scratches ($30.59\%$) were predicted as normal clean wafers (vs. 37 in Custom CNN Exp 1 and 36 in Exp 2). ResNet-18 missed thin, single-die scratch streaks.
   - **`Loc` Escapes:** 97 out of 513 localized clusters ($18.91\%$) were predicted as clean (vs. 70 in Exp 1 and 65 in Exp 2).
   - **`Edge-Loc` Escapes:** 135 out of 741 edge defects ($18.22\%$) escaped to clean (vs. 88 in Exp 1).
3. **Defect-to-Defect Boundary Confusions:**
   - `Loc` $\longleftrightarrow$ `Edge-Loc`: 28 `Loc` wafers were classified as `Edge-Loc`, and 20 `Edge-Loc` wafers were classified as `Loc`.
   - `Donut` $\longleftrightarrow$ `Center`: 7 `Donut` wafers were misclassified as `Center` due to concentric circular symmetry overlap.

---

## 8. Overfitting Observations & Domain Analysis

### 8.1 Empirical Overfitting Profile
- **Capacity Discrepancy:** ResNet-18 contains **11,174,857 parameters**, which is $21.5\times$ larger than `CustomWaferCNN` (520,681 parameters).
- **Training Loss Trajectory:** Driven down from $0.4579$ (Epoch 1) to **$0.000746$** (Epoch 30), indicating virtually total training set memorization.
- **Validation Loss Trajectory:** Reached a global minimum of **$0.1168$ at Epoch 6**, after which validation loss steadily drifted upward to **$0.3275$ at Epoch 23**, while validation macro F1 peaked at **0.8611** before leveling off.

### 8.2 Why Custom CNN Outperforms Pretrained ResNet-18
1. **Early Spatial Downsampling:**
   - ResNet-18 begins with `conv1` ($7 \times 7$, stride 2) followed immediately by `maxpool` ($3 \times 3$, stride 2). This contracts the wafer map from $64 \times 64 \to 32 \times 32 \to \mathbf{16 \times 16}$ **before the first residual block**.
   - For a discrete $64 \times 64$ die matrix, a scratch is often a 1-pixel wide discontinuous trace, and a localized cluster spans only a 3–5 die patch. Downsampling by $4\times$ in the input stem destroys local topological connectivity.
   - In contrast, `CustomWaferCNN` uses $3 \times 3$ stride-1 convolutions with progressive single $2 \times 2$ pooling stages ($64 \to 32 \to 16 \to 8 \to 4$), preserving fine die grid structures.
2. **Domain Disparity (Natural Images vs. Discrete Lattices):**
   - ImageNet features are optimized for multi-color photorealistic objects, gradients, and continuous lighting.
   - Semiconductor wafer maps are strictly ternary discrete grids ($\{0: \text{exterior}, 1: \text{normal}, 2: \text{defective}\}$) with rotational and radial manufacturing symmetry. Higher-order natural texture filters do not transfer naturally to discrete die defect geometries.

---

## 9. Hardware & Runtime Profiling (NVIDIA RTX 2050 4GB)

- **Total Wall Duration:** 94.84 minutes ($5,690.3\text{ s}$)
- **Active Training Duration:** 94.74 minutes ($5,684.5\text{ s}$)
- **Average Epoch Duration:** **189.5 seconds / epoch**
  - Stage 1 Warmup (Frozen backbone): **~61.0 s / epoch**
  - Stage 2 Fine-Tuning (Unfrozen backbone): **~203.7 s / epoch**
- **VRAM Utilization Profile:**
  - Peak PyTorch Active Allocated VRAM: **233.5 MB**
  - Peak PyTorch Reserved VRAM: **268.0 MB**
  - Peak Process GPU Memory (`nvidia-smi`): **386.0 MB** (less than 10% of the 4,096 MB VRAM capacity)
  - Average GPU Utilization: **47.6%**

---

## 10. Local Artifacts & Reproducibility Paths

| Artifact Type | Path | Purpose |
|:---|:---|:---|
| **Best Model Checkpoint** | [`models/checkpoints/resnet18_pretrained_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/resnet18_pretrained_best.pt) | Saved best checkpoint (Epoch 23, Macro F1: 0.8611, 44.7 MB) |
| **Training Curves Plot** | [`data/interim/figures/resnet18_pretrained_training_curves.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_pretrained_training_curves.png) | Epoch-wise train/val loss, macro precision, recall, and F1 |
| **Confusion Matrix Plot** | [`data/interim/figures/resnet18_pretrained_confusion_matrix.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_pretrained_confusion_matrix.png) | Normalized and raw confusion matrix on validation set |
| **Complete Metrics JSON** | [`data/interim/figures/resnet18_pretrained_metrics.json`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_pretrained_metrics.json) | Complete numerical history, per-class metrics, timing, and VRAM |
| **Training Runner Script** | [`scripts/train_resnet18_pretrained.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/train_resnet18_pretrained.py) | Fully reproducible training script with multi-stage schedule |
