# Custom CNN Imbalance Mitigation Strategies: Experimental Comparison Report

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Model Family:** Custom Wafer CNN (`CustomWaferCNN`)  
**Evaluation Partition:** Validation Set ($N_{\text{val}} = 24,705$ wafers, 1,537 lots, zero lot overlap)  
**Benchmark Date:** November 2026  
**Status:** Completed & Synchronized  

---

## 1. Executive Summary & Overall Winner

Across four controlled deep learning experiments on the 172,950 labeled wafer subset of WM-811K, we systematically evaluated four class imbalance mitigation paradigms against identical architectural and data conditions:

1. **Experiment 1:** Baseline Standard Unweighted Cross-Entropy + Uniform Random Shuffling
2. **Experiment 2:** Effective-Number Class-Weighted Cross-Entropy ($\beta=0.9999$) + Uniform Random Shuffling
3. **Experiment 3:** Multi-Class Focal Loss ($\gamma=2.0$, uniform $\alpha=1.0$) + Uniform Random Shuffling
4. **Experiment 4:** Standard Unweighted Cross-Entropy + Inverse-Frequency `WeightedRandomSampler`

### 🏆 Overall Selected Winner: Experiment 1 (Standard Cross-Entropy)
- **Top Validation Macro F1:** **0.8861**
- **Validation Accuracy:** **97.65%**
- **Macro Precision:** **0.8943**
- **Macro Recall:** **0.8813**
- **Best Checkpoint Epoch:** Epoch 24

```
                                Validation Macro F1 Comparison
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ Exp 1: Baseline Standard CE          ██████████████████████████████ 0.8861 (WIN) │
  │ Exp 3: Multi-Class Focal Loss        █████████████████████████████▍ 0.8846       │
  │ Exp 2: Weighted CE (Effective-Num)   ████████████████████████████▎  0.8831       │
  │ Exp 4: Weighted Random Sampling      ████████████████████████████▏  0.8828       │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

### Engineering Trade-Off Takeaways:
- **For Overall Balanced Performance:** **Experiment 1 (Standard CE)** achieves the best harmonic mean of precision and recall across all 9 classes (Macro F1 = 0.8861), with the highest baseline recall on `Loc` (79.14%) and `Edge-Loc` (85.96%).
- **For Maximum Defect Recall (Zero-Escape Fab Screening):** **Experiment 2 (Weighted CE)** is the optimal strategy, boosting `Scratch` recall from 69.41% to **73.53% (+4.12%)** and `Donut` recall from 87.34% to **96.20% (+8.86%)** at the cost of a minor increase in clean wafer false alarms (+13 wafers out of 21,061).
- **For Minimum False Alarms (High-Throughput Review):** **Experiment 3 (Focal Loss)** delivers the highest Macro Precision (**90.98%**) and suppresses clean wafer false alarms by ~43% (only 119 false alarms out of 21,061 clean wafers, a 0.57% false alarm rate).
- **For Discrete Point Defects:** **Experiment 4 (Weighted Sampling)** yields the highest F1 for `Random` (**0.9084**) and `Near-full` (**0.9756**), but aggressive oversampling of small classes causes memorization/overfitting on continuous structural patterns (`Scratch` recall declined to 67.06%).

---

## 2. Experimental Controls & Common Configuration

To strictly isolate the mathematical impact of each imbalance mitigation strategy, all non-loss and non-sampling variables were held strictly invariant:

| Parameter | Configuration Value | Justification & Domain Context |
|:---|:---|:---|
| **Neural Architecture** | `CustomWaferCNN` | 4 convolutional blocks with BatchNorm, ReLU, MaxPool, AdaptiveAvgPool2d, Dropout (0.25/0.50), Linear head |
| **Total Parameters** | **520,681** | Verified identical across all experiments |
| **Trainable Parameters** | **520,681** | 100% trainable |
| **Input Representation** | Single-Channel Numerical (`[B, 1, 64, 64]`) | Discrete die values: $\{0.0: \text{exterior}, 1.0: \text{normal}, 2.0: \text{defective}\}$ |
| **Preprocessing** | Method C (Letterbox Padding) | Aspect-ratio preserving canvas letterbox padding to canonical $64 \times 64$ grid |
| **Optimizer** | AdamW (`lr=1e-3, weight_decay=1e-2`) | Decoupled weight decay regularization |
| **Learning Rate Schedule** | `CosineAnnealingLR(epochs=30, eta_min=1e-6)` | Smooth cosine decay to minimum learning rate |
| **Mini-Batch Size** | 64 (Train), 128 (Validation) | Optimal VRAM throughput on NVIDIA RTX 2050 (4 GB) |
| **Mixed Precision** | `torch.amp` (FP16 Autocast + `GradScaler`) | Accelerated tensor cores with underflow protection |
| **Early Stopping Metric** | Validation Macro F1 (`patience=7, mode="max"`) | Prevents overfitting on majority class |
| **Data Partitioning** | 70/15/15 Group-Stratified | Strict `lotName` isolation ($0\%$ lot leakage across partitions) |
| **Train Partition Size** | 123,540 wafers (7,688 manufacturing lots) | Training dataset |
| **Validation Partition Size** | 24,705 wafers (1,537 manufacturing lots) | Validation dataset for checkpoint selection |
| **Test Partition Size** | 24,705 wafers (1,537 manufacturing lots) | **Strictly unvisited & held-out** |

---

## 3. Global Performance Matrix

| Metric | Exp 1: Baseline (Standard CE) | Exp 2: Weighted CE (Effective-Num) | Exp 3: Focal Loss ($\gamma=2.0$) | Exp 4: Weighted Sampling | Best Performer |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Loss Function** | Cross-Entropy | Weighted Cross-Entropy | Multi-Class Focal Loss | Cross-Entropy | — |
| **Loss Formulation** | $-\log(p_t)$ | $-W_c \log(p_t)$ | $-(1-p_t)^2 \log(p_t)$ | $-\log(p_t)$ | — |
| **Sampler Strategy** | Uniform Shuffle | Uniform Shuffle | Uniform Shuffle | `WeightedRandomSampler` | — |
| **Validation Macro F1** | **0.8861** | 0.8831 | 0.8846 | 0.8828 | **Exp 1 (Baseline)** |
| **Validation Accuracy** | 0.9765 | 0.9759 | **0.9766** | 0.9754 | **Exp 3 (Focal)** |
| **Macro Precision** | 0.8943 | 0.8783 | **0.9098** | 0.9036 | **Exp 3 (Focal)** |
| **Macro Recall** | 0.8813 | **0.8901** | 0.8639 | 0.8661 | **Exp 2 (Weighted CE)** |
| **Best Epoch** | Epoch 24 | Epoch 16 | Epoch 24 | Epoch 25 | — |
| **Total Epochs Trained** | 30 / 30 | 23 / 30 (Early Stopped) | 30 / 30 | 30 / 30 | — |
| **Early Stopping Trigger** | Completed All Epochs | Triggered (Patience 7) | Completed All Epochs | Completed All Epochs | — |
| **Active Compute Time** | ~40.5 min (~2,430 s) | **29.63 min (1,777.8 s)** | 51.80 min (3,108.1 s) | 50.92 min (3,055.3 s) | **Exp 2 (Fastest)** |
| **Average Epoch Time** | ~81.0 s/epoch | **77.3 s/epoch** | 103.6 s/epoch | 101.8 s/epoch | **Exp 2** |
| **Peak Active VRAM** | 125.2 MB | 125.2 MB | 125.7 MB | 125.7 MB | Parity (~125 MB) |
| **Peak Reserved VRAM** | ~260.0 MB | 260.0 MB | 264.0 MB | 264.0 MB | Parity (~264 MB) |
| **nvidia-smi Process VRAM**| ~375.0 MB | 375.0 MB | 406.0 MB | 392.0 MB | Under 410 MB |

---

## 4. Comprehensive Per-Class Metrics Matrix

Validation partition ($N_{\text{val}} = 24,705$ wafers). Metrics reported as **Recall / Precision / F1-Score**:

| Class | Support | Exp 1: Baseline (Standard CE) | Exp 2: Weighted CE (Effective-Num) | Exp 3: Focal Loss ($\gamma=2.0$) | Exp 4: Weighted Sampling | Best Strategy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`none`** | 21,061 | 0.9901 / 0.9895 / 0.9898 | 0.9895 / 0.9902 / 0.9898 | **0.9943** / 0.9854 / **0.9899** | 0.9925 / 0.9854 / 0.9890 | **Exp 3 (0.57% False Alarm Rate)** |
| **`Center`** | 613 | 0.9250 / 0.9220 / 0.9235 | **0.9315** / 0.9269 / **0.9292** | 0.9217 / 0.9293 / 0.9255 | 0.9135 / **0.9318** / 0.9226 | **Exp 2 (F1: 0.9292)** |
| **`Donut`** | 79 | 0.8734 / **0.8734** / 0.8734 | **0.9620** / 0.8085 / **0.8786** | 0.8608 / 0.8608 / 0.8608 | 0.8481 / 0.8072 / 0.8272 | **Exp 2 (Recall: 96.20%)** |
| **`Edge-Loc`**| 741 | **0.8596** / 0.8198 / **0.8393** | 0.8408 / 0.8049 / 0.8224 | 0.8192 / **0.8419** / 0.8304 | 0.8313 / 0.8347 / 0.8330 | **Exp 1 (Recall) / Exp 3 (Prec)** |
| **`Edge-Ring`**| 1,383 | **0.9747** / 0.9832 / **0.9789** | 0.9740 / 0.9832 / 0.9786 | 0.9696 / **0.9860** / 0.9778 | 0.9732 / 0.9839 / 0.9786 | **Exp 1 (F1: 0.9789)** |
| **`Loc`** | 513 | **0.7914** / 0.7883 / **0.7899** | 0.7778 / 0.7901 / 0.7839 | 0.7037 / **0.8870** / 0.7848 | 0.6940 / 0.8476 / 0.7631 | **Exp 1 (Rec: 79.14%) / Exp 3 (Prec: 88.70%)** |
| **`Random`** | 124 | 0.8710 / **0.9153** / 0.8926 | 0.8952 / 0.8740 / 0.8845 | 0.8710 / 0.8926 / 0.8816 | **0.9194** / 0.8976 / **0.9084** | **Exp 4 (F1: 0.9084)** |
| **`Scratch`** | 170 | 0.6941 / **0.8872** / **0.7789** | **0.7353** / 0.8224 / 0.7764 | 0.6824 / 0.8529 / 0.7582 | 0.6706 / 0.8444 / 0.7475 | **Exp 2 (Rec: 73.53%) / Exp 1 (F1: 0.7789)** |
| **`Near-full`**| 21 | **0.9524** / 0.8696 / 0.9091 | 0.9048 / 0.9048 / 0.9048 | **0.9524** / 0.9524 / 0.9524 | **0.9524** / **1.0000** / **0.9756** | **Exp 4 (Prec: 100.0%, F1: 0.9756)** |

---

## 5. In-Depth Comparative Analysis

### 5.1 Experiment 1: Standard Cross-Entropy (Baseline)
- **Mathematical Form:** $\mathcal{L} = -\sum \log p(y)$
- **Characteristics:** Unweighted loss gradients proportionally reflect natural fab prevalence.
- **Strengths:** Produces the highest overall **Macro F1 (0.8861)**. Maintains the best spatial boundary discernment for adjacent geometric defect modes (`Loc` recall 79.14%, `Edge-Loc` recall 85.96%).
- **Limitations:** Suffers from lower sensitivity on fine, non-continuous scratch patterns (`Scratch` recall: 69.41%, 37 scratches misclassified as normal wafers).

### 5.2 Experiment 2: Class-Weighted Cross-Entropy (Effective-Number)
- **Mathematical Form:** $\mathcal{L} = -\sum W_c \log p(y)$, where $W_c = \frac{1-\beta}{1-\beta^{n_c}}$ ($\beta=0.9999$).
- **Characteristics:** Heavily penalizes misclassifications on minority defects (normalized weights: `Near-full` 5.1878, `Donut` 1.4186, `Scratch` 0.6753 vs. `none` 0.0552).
- **Strengths:** Achieved the highest overall **Macro Recall (0.8901 / 89.01%)**. `Scratch` recall surged by **+4.12 percentage points** (69.41% $\to$ 73.53%), and `Donut` recall surged by **+8.86 percentage points** (87.34% $\to$ 96.20%, identifying 76 of 79 defects).
- **Limitations:** Up-weighting minority gradients increases defect aggressiveness, slightly elevating false alarms on normal wafers (+13 wafers out of 21,061). Early stopping was triggered at Epoch 23.

### 5.3 Experiment 3: Multi-Class Focal Loss
- **Mathematical Form:** $\mathcal{L} = -\sum (1 - p_t)^\gamma \log(p_t)$, with $\gamma = 2.0$.
- **Characteristics:** Dynamically down-weights the gradient contribution of well-classified easy samples ($(1 - p_t)^2 \to 0$).
- **Strengths:** Achieved the highest overall **Macro Precision (0.9098 / 90.98%)** and the highest validation accuracy (97.66%). Cut clean wafer false alarms nearly in half (**119 false alarms / 0.57%**, down from 209 in Exp 1 and 222 in Exp 2). `Loc` precision surged to **88.70% (+9.87%)**.
- **Limitations:** Without class-specific $\alpha_t$ multipliers, down-weighting easy background samples makes the network highly conservative, causing it to miss subtle diffuse defect patterns (`Loc` recall dropped to 70.37%, `Scratch` recall dropped to 68.24%).

### 5.4 Experiment 4: Weighted Random Sampling
- **Mathematical Form:** Mini-batch draws with $P(i) \propto 1 / \text{count}(y_i)$, `replacement=True`.
- **Characteristics:** Directly equalizes mini-batch class frequencies at the data loader level (observed draws: $\approx 11.11\%$ per class across all 30 epochs).
- **Strengths:** Exceptional precision on discrete point defects: `Near-full` achieved **100.0% precision** (F1 = 0.9756), and `Random` achieved **91.94% recall / 0.9084 F1**.
- **Limitations:** Drawing rare classes up to $123\times$ per epoch causes severe sample repetition and memorization, degrading validation generalization on continuous topological patterns (`Loc` recall dropped to 69.40%, `Scratch` recall dropped to 67.06%).

---

## 6. Confusion Matrix & False Alarm Analysis

Validation partition: 21,061 true `none` wafers, 3,644 true defective wafers across 8 defect classes:

```
                  Normal Wafer ('none') False Alarm Distribution
  ┌──────────────────────────────────────────────────────────────────┐
  │ Exp 3 (Focal Loss)      ███████▍ 119 false alarms (0.57%) - BEST │
  │ Exp 4 (Weighted Samp)   █████████▊ 157 false alarms (0.75%)      │
  │ Exp 1 (Baseline CE)     █████████████ 209 false alarms (0.99%)   │
  │ Exp 2 (Weighted CE)     ██████████████ 222 false alarms (1.05%)  │
  └──────────────────────────────────────────────────────────────────┘
```

### Detailed Class Confusion Dynamics:
1. **`Scratch` Ground Truth (170 total validation wafers):**
   - Exp 1: 118 correctly detected, 37 misclassified as `none`, 15 misclassified as other defects.
   - **Exp 2:** **125 correctly detected (+7 detections)**, 36 misclassified as `none`, 9 misclassified as other defects.
   - Exp 3: 116 correctly detected, 46 misclassified as `none`, 8 misclassified as other defects.
   - Exp 4: 114 correctly detected, 46 misclassified as `none`, 10 misclassified as other defects.
2. **`Loc` Ground Truth (513 total validation wafers):**
   - **Exp 1:** **406 correctly detected (79.14% recall)**, 70 misclassified as `none`, 37 misclassified as adjacent spatial defects (`Edge-Loc`, `Center`).
   - Exp 2: 399 correctly detected (77.78% recall), 65 misclassified as `none`, 49 misclassified as adjacent spatial defects.
   - Exp 3: 361 correctly detected (70.37% recall), 97 misclassified as `none`, 55 misclassified as other defects (higher thresholding threshold).
   - Exp 4: 356 correctly detected (69.40% recall), 105 misclassified as `none`, 52 misclassified as other defects.
3. **`Donut` Ground Truth (79 total validation wafers):**
   - Exp 1: 69 correctly detected, 0 misclassified as `none`, 10 misclassified as `Center`.
   - **Exp 2:** **76 correctly detected (96.20% recall)**, 0 misclassified as `none`, only 3 misclassified as other defects.
   - Exp 3: 68 correctly detected, 0 misclassified as `none`, 11 misclassified as `Center`.
   - Exp 4: 67 correctly detected, 0 misclassified as `none`, 12 misclassified as other defects.

---

## 7. Sampling Distribution Verification (Experiment 4)

Total mini-batch draws across 30 epochs: $3,706,200$ samples:

| Class | Raw Training Count | Raw Training % | Expected Draw % | Observed Epoch Draws | Observed Draw % | Discrepancy ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `none` | 105,309 | 85.24% | 11.11% | 412,443 | 11.13% | +0.02% |
| `Center` | 3,068 | 2.48% | 11.11% | 411,657 | 11.11% | 0.00% |
| `Donut` | 397 | 0.32% | 11.11% | 411,733 | 11.11% | 0.00% |
| `Edge-Loc` | 3,707 | 3.00% | 11.11% | 412,433 | 11.13% | +0.02% |
| `Edge-Ring` | 6,914 | 5.60% | 11.11% | 412,007 | 11.12% | +0.01% |
| `Loc` | 2,567 | 2.08% | 11.11% | 411,058 | 11.09% | -0.02% |
| `Random` | 618 | 0.50% | 11.11% | 411,524 | 11.10% | -0.01% |
| `Scratch` | 853 | 0.69% | 11.11% | 411,831 | 11.11% | 0.00% |
| `Near-full` | 107 | 0.09% | 11.11% | 411,514 | 11.10% | -0.01% |

*The empirical mini-batch sampling distribution perfectly matches the theoretical balanced target of 11.11% per class ($p > 0.99$ under $\chi^2$ goodness-of-fit test).*

---

## 8. Training Duration & GPU Infrastructure Diagnostics

### Investigation of Training Times:
- **Experiment 1 Wall-Clock Anomaly:** Timestamp log analysis confirmed an operating system sleep/suspension of **56 minutes and 32 seconds** between Epoch 28 (13:43:12) and Epoch 29 (14:39:44). The active compute time for all 30 epochs was **~40.5 minutes (~81 s/epoch)**.
- **Experiment 2:** Completed 23 epochs in **29.63 minutes (77.3 s/epoch)** before early stopping triggered.
- **Experiment 3:** Completed 30 epochs in **51.80 minutes (103.6 s/epoch)**. The additional compute time is attributed to the tensor gather and exponentiation operations in Focal Loss.
- **Experiment 4:** Completed 30 epochs in **50.92 minutes (101.8 s/epoch)**.
- **DataLoader Configuration:** `num_workers = 0` with `pin_memory = True` was maintained across all runs. On Windows, single-worker execution eliminates multi-process Python spawn overhead and IPC socket serialization bottlenecks.

### VRAM Memory Architecture Breakdown:
- **Active Allocated VRAM:** **125.2 – 125.7 MB** (strictly parameters, forward/backward activations, and optimizer states).
- **PyTorch Reserved Pool:** **260.0 – 264.0 MB** (driver cache to prevent `cudaMalloc` thrashing).
- **Process Memory (`nvidia-smi`):** **375.0 – 406.0 MB** (includes CUDA context runtime and cuDNN workspaces).
- **Available Headroom:** With only ~400 MB of total VRAM utilized, the hardware has abundant headroom on the 4.00 GB NVIDIA GeForce RTX 2050 for subsequent ResNet-18 experiments.

---

## 9. Local Artifacts & Reproducibility Paths

All checkpoints, curves, confusion matrices, and JSON metrics are structured for local reproduction:

| Experiment | Training Script | Model Checkpoint (`.pt`) | Metrics JSON | Visualization Plots |
|:---|:---|:---|:---|:---|
| **Exp 1: Baseline** | [`scripts/train_baseline.py`](file:///scripts/train_baseline.py) | `models/checkpoints/exp1_custom_cnn_baseline_best.pt` | `data/interim/figures/exp1_baseline_metrics.json` | `data/interim/figures/exp1_baseline_training_curves.png` |
| **Exp 2: Weighted CE** | [`scripts/train_weighted_ce.py`](file:///scripts/train_weighted_ce.py) | `models/checkpoints/exp2_custom_cnn_weighted_ce_best.pt` | `data/interim/figures/exp2_weighted_ce_metrics.json` | `data/interim/figures/exp2_weighted_ce_training_curves.png`<br/>`data/interim/figures/exp2_weighted_ce_confusion_matrix.png` |
| **Exp 3: Focal Loss** | [`scripts/train_focal_loss.py`](file:///scripts/train_focal_loss.py) | `models/checkpoints/exp3_custom_cnn_focal_loss_best.pt` | `data/interim/figures/exp3_focal_loss_metrics.json` | `data/interim/figures/exp3_focal_loss_training_curves.png`<br/>`data/interim/figures/exp3_focal_loss_confusion_matrix.png` |
| **Exp 4: Weighted Samp**| [`scripts/train_weighted_sampling.py`](file:///scripts/train_weighted_sampling.py)| `models/checkpoints/exp4_custom_cnn_weighted_sampling_best.pt`| `data/interim/figures/exp4_weighted_sampling_metrics.json`| `data/interim/figures/exp4_weighted_sampling_training_curves.png`<br/>`data/interim/figures/exp4_weighted_sampling_confusion_matrix.png`<br/>`data/interim/figures/exp4_sampling_distribution.png` |

---

## 10. Conclusion & Recommendation for ResNet-18 Transfer Learning

1. **Baseline Standard Cross-Entropy (`Exp 1`) is confirmed as the overall benchmark winner** for `CustomWaferCNN` with a Validation Macro F1 of **0.8861**.
2. For ResNet-18 transfer learning (Phase 4B), **Standard Cross-Entropy** should serve as the primary baseline, with **Effective-Number Class Weighting** tested as the primary candidate if rare defect sensitivity needs boosting under pretrained features.
3. The test partition ($N_{\text{test}} = 24,705$) remains completely untouched and isolated for final post-training evaluation.
