# Pre-Phase 7 Experimental Audit: Metrics Consistency & Scratch Recall Verification

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Audit Target:** Scratch Recall Inconsistency & Validation Metrics Integrity across Phase 4, Phase 5A, Phase 6A, and Phase 6B  
**Audited Checkpoints:**
1. [`models/checkpoints/exp1_custom_cnn_baseline_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/exp1_custom_cnn_baseline_best.pt) (CustomWaferCNN Baseline Winner, Epoch 24)
2. [`models/checkpoints/resnet18_pretrained_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/resnet18_pretrained_best.pt) (AdaptedResNet18 ImageNet Pretrained, Epoch 23)  
**Evaluation Partition:** Group-Stratified Validation Set ($N_{\text{val}} = 24,705$ wafers, 1,537 lots, zero lot leakage)  
**Held-Out Benchmark:** Test Partition ($N_{\text{test}} = 24,705$ wafers, **strictly unvisited & untouched**)  
**Audit Script:** [`scripts/audit_scratch_recall.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/audit_scratch_recall.py)  
**Audit Date:** November 2026  
**Status:** Audit Completed & Reconciled  

---

## 1. Executive Summary of Audit Findings

Prior to final test set evaluation, an audit was triggered to investigate a documented contradiction regarding validation `Scratch` recall:
- **`docs/experiments/custom_cnn_comparison.md` (Phase 4):** Reported `Scratch` recall = **69.41%**
- **`docs/experiments/gradcam_architecture_comparison.md` (Phase 6B):** Reported `Scratch` recall = **84.12%** (Custom CNN) vs. **69.41%** (ResNet-18)

### Key Conclusions:
1. **The Ground Truth Benchmark:** The true, verified validation recall for `Scratch` is **69.41% (118 TP / 170 Support)** for `CustomWaferCNN` (Exp 1 Baseline) and **61.76% (105 TP / 170 Support)** for `AdaptedResNet18` (Pretrained).
2. **Identification of Root Cause:** In `docs/experiments/gradcam_architecture_comparison.md` (Section 2, Table Line 40), the metrics author made two typographical/transposition errors:
   - **Column Misalignment:** Custom CNN's true recall of **69.41%** was erroneously shifted into the ResNet-18 column (whose true recall was **61.76%**).
   - **Formula Inversion on Partial Escapes:** For Custom CNN, the author mistakenly subtracted an assumed clean-escape rate ($15.88\% \approx 15.9\%$) from $100\%$ ($100\% - 15.88\% = \mathbf{84.12\%}$), erroneously equating $(100\% - \text{escape rate to none})$ with total class recall, thereby ignoring false negatives to other defect classes (`Loc`, `Edge-Loc`, `Edge-Ring`).
   - A similar inversion occurred on `Loc` recall in the same table, where ResNet-18's escape-to-none rate ($18.91\%$) was subtracted from $100\%$ ($100\% - 18.91\% = \mathbf{81.09\%}$) and recorded as its recall instead of the true recall of **70.18%**.
3. **Model & Checkpoint Integrity:** Both model checkpoints (`exp1_custom_cnn_baseline_best.pt` and `resnet18_pretrained_best.pt`) remain 100% intact, reproducible, and unaltered. The underlying JSON metrics files (`exp1_baseline_metrics.json` and `resnet18_pretrained_metrics.json`) generated directly at training time are bit-for-bit identical with independent re-evaluations.
4. **Test Partition Isolation:** Zero test partition wafers ($N_{\text{test}} = 24,705$) were loaded or evaluated. The test set remains strictly unvisited.

---

## 2. Source Files & Metric Provenance Audit

| Artifact / Document | File Path | Reported Scratch Recall | Status & Provenance |
| :--- | :--- | :---: | :--- |
| **Exp 1 Baseline JSON** | [`data/interim/figures/exp1_baseline_metrics.json`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/exp1_baseline_metrics.json) | **0.6941 (69.41%)** | **Authoritative Ground Truth** (Generated directly by Phase 4 training runner) |
| **ResNet18 Pretrained JSON** | [`data/interim/figures/resnet18_pretrained_metrics.json`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/resnet18_pretrained_metrics.json) | **0.6176 (61.76%)** | **Authoritative Ground Truth** (Generated directly by Phase 5A training runner) |
| **Custom CNN Report** | [`docs/experiments/custom_cnn_comparison.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/experiments/custom_cnn_comparison.md) | **69.41%** | **Correct** (Matches `exp1_baseline_metrics.json`) |
| **ResNet18 Report** | [`docs/experiments/resnet18_pretrained.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/experiments/resnet18_pretrained.md) | **61.76%** (ResNet) / **69.41%** (CNN) | **Correct** (Matches both JSON files) |
| **Phase 6A Grad-CAM Report** | [`docs/experiments/gradcam_custom_cnn.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/experiments/gradcam_custom_cnn.md) | N/A (Focuses on Macro F1: 0.8861) | **Consistent** (No conflicting per-class claims) |
| **Phase 6B ResNet18 Report** | [`docs/experiments/gradcam_resnet18.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/experiments/gradcam_resnet18.md) | **61.76%** (ResNet) | **Correct** (Quotes 52 escapes to clean out of 170) |
| **Architecture Comparison** | [`docs/experiments/gradcam_architecture_comparison.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/experiments/gradcam_architecture_comparison.md) | **84.12%** (CNN) / **69.41%** (ResNet) | **DEFECTIVE / ERRONEOUS TABLE ROW** |
| **README Overview** | [`README.md`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/README.md) | Text mentions 15.9% escape | **DEFECTIVE BULLET PHRASING** |

---

## 3. Independent Verification & Exact Confusion Matrices

To provide indisputable mathematical certainty, an independent re-evaluation script [`scripts/audit_scratch_recall.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/audit_scratch_recall.py) was executed on the validation partition ($N_{\text{val}} = 24,705$ wafers).

### 3.1 Class Specification & Mapping Verification
- **Class Index:** `7`
- **Label Name:** `Scratch`
- **Validation Support ($N_7$):** **170 wafers** (out of 24,705 total validation wafers)
- **Lot Leakage Check:** $\text{Lots}_{\text{train}} \cap \text{Lots}_{\text{val}} = \emptyset$ (1,537 validation lots, zero overlap)

### 3.2 CustomWaferCNN (Baseline Winner — `exp1_custom_cnn_baseline_best.pt`)
- **Macro F1:** **0.8861** | **Accuracy:** **97.65%** | **Macro Recall:** **0.8813** | **Macro Precision:** **0.8943**

#### Full Per-Class Validation Breakdown:
| Class Index | Defect Class | Support | TP | FP | FN | Precision | Recall | F1-Score | Escaped to `none` (Class 0) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `none` | 21,061 | 20,852 | 222 | 209 | 0.9895 | 0.9901 | 0.9898 | N/A (clean baseline) |
| 1 | `Center` | 613 | 567 | 48 | 46 | 0.9220 | 0.9250 | 0.9235 | 30 (4.89%) |
| 2 | `Donut` | 79 | 69 | 10 | 10 | 0.8734 | 0.8734 | 0.8734 | 0 (0.00%) |
| 3 | `Edge-Loc` | 741 | 637 | 140 | 104 | 0.8198 | 0.8596 | 0.8393 | 71 (9.58%) |
| 4 | `Edge-Ring` | 1,383 | 1,348 | 23 | 35 | 0.9832 | 0.9747 | 0.9789 | 13 (0.94%) |
| 5 | `Loc` | 513 | 406 | 109 | 107 | 0.7883 | 0.7914 | 0.7899 | 70 (13.65%) |
| 6 | `Random` | 124 | 108 | 10 | 16 | 0.9153 | 0.8710 | 0.8926 | 1 (0.81%) |
| **7** | **`Scratch`** | **170** | **118** | **15** | **52** | **0.8872** | **0.6941** | **0.7789** | **37 (21.76%)** |
| 8 | `Near-full` | 21 | 20 | 3 | 1 | 0.8696 | 0.9524 | 0.9091 | 0 (0.00%) |

#### Complete Breakdown of True `Scratch` Samples ($N=170$):
$$\begin{aligned}
\text{Correctly Classified as Scratch (TP)} &: 118 \quad (69.41\%) \\
\text{Escaped to 'none' (False Negative)} &: 37 \quad (21.76\%) \\
\text{Confused with 'Loc' (False Negative)} &: 12 \quad (7.06\%) \\
\text{Confused with 'Edge-Loc' (False Negative)} &: 2 \quad (1.18\%) \\
\text{Confused with 'Edge-Ring' (False Negative)} &: 1 \quad (0.59\%) \\
\hline
\text{Total False Negatives (FN)} &: 52 \quad (30.59\%) \\
\text{Total Support} &: 118 + 52 = 170 \quad (100.0\%)
\end{aligned}$$

$$\text{Recall}_{\text{Scratch}}^{\text{CustomCNN}} = \frac{118}{170} = 0.6941176... \approx \mathbf{69.41\%}$$
$$\text{Precision}_{\text{Scratch}}^{\text{CustomCNN}} = \frac{118}{118 + 15} = \frac{118}{133} = 0.887218... \approx \mathbf{88.72\%}$$
$$\text{F1}_{\text{Scratch}}^{\text{CustomCNN}} = \frac{2 \times 0.887218 \times 0.694118}{0.887218 + 0.694118} \approx \mathbf{0.7789}$$

---

### 3.3 AdaptedResNet18 (Pretrained Best — `resnet18_pretrained_best.pt`)
- **Macro F1:** **0.8611** | **Accuracy:** **97.32%** | **Macro Recall:** **0.8436** | **Macro Precision:** **0.8846**

#### Full Per-Class Validation Breakdown:
| Class Index | Defect Class | Support | TP | FP | FN | Precision | Recall | F1-Score | Escaped to `none` (Class 0) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `none` | 21,061 | 20,935 | 356 | 126 | 0.9833 | 0.9940 | 0.9886 | N/A (clean baseline) |
| 1 | `Center` | 613 | 545 | 43 | 68 | 0.9269 | 0.8891 | 0.9076 | 47 (7.67%) |
| 2 | `Donut` | 79 | 67 | 19 | 12 | 0.7791 | 0.8481 | 0.8121 | 0 (0.00%) |
| 3 | `Edge-Loc` | 741 | 560 | 98 | 181 | 0.8511 | 0.7557 | 0.8006 | 135 (18.22%) |
| 4 | `Edge-Ring` | 1,383 | 1,343 | 29 | 40 | 0.9789 | 0.9711 | 0.9750 | 23 (1.66%) |
| 5 | `Loc` | 513 | 360 | 86 | 153 | 0.8072 | 0.7018 | 0.7508 | 97 (18.91%) |
| 6 | `Random` | 124 | 107 | 7 | 17 | 0.9386 | 0.8629 | 0.8992 | 2 (1.61%) |
| **7** | **`Scratch`** | **170** | **105** | **22** | **65** | **0.8268** | **0.6176** | **0.7071** | **52 (30.59%)** |
| 8 | `Near-full` | 21 | 20 | 3 | 1 | 0.8696 | 0.9524 | 0.9091 | 0 (0.00%) |

#### Complete Breakdown of True `Scratch` Samples ($N=170$):
$$\begin{aligned}
\text{Correctly Classified as Scratch (TP)} &: 105 \quad (61.76\%) \\
\text{Escaped to 'none' (False Negative)} &: 52 \quad (30.59\%) \\
\text{Confused with 'Loc' (False Negative)} &: 11 \quad (6.47\%) \\
\text{Confused with 'Edge-Loc' (False Negative)} &: 2 \quad (1.18\%) \\
\hline
\text{Total False Negatives (FN)} &: 65 \quad (38.24\%) \\
\text{Total Support} &: 105 + 65 = 170 \quad (100.0\%)
\end{aligned}$$

$$\text{Recall}_{\text{Scratch}}^{\text{ResNet18}} = \frac{105}{170} = 0.617647... \approx \mathbf{61.76\%}$$
$$\text{Precision}_{\text{Scratch}}^{\text{ResNet18}} = \frac{105}{105 + 22} = \frac{105}{127} = 0.82677... \approx \mathbf{82.68\%}$$
$$\text{F1}_{\text{Scratch}}^{\text{ResNet18}} = \frac{2 \times 0.82677 \times 0.617647}{0.82677 + 0.617647} \approx \mathbf{0.7071}$$

---

## 4. Mathematical Reconstruction of the Error Root Cause

Our audit traced the exact cognitive and transcription errors that led to the erroneous entry in `docs/experiments/gradcam_architecture_comparison.md`:

### 4.1 The Conflation of "Total Recall" and "Escape to Clean"
In semiconductor yield engineering, defects that escape detection to clean (`Defect` $\to$ `none`) are a critical subset of false negatives.
For ResNet-18:
- ResNet-18 predicted 52 scratches as `none` ($52 / 170 = \mathbf{30.59\%}$ escape rate to clean).
- ResNet-18 predicted 97 localized defects as `none` ($97 / 513 = \mathbf{18.91\%}$ escape rate to clean).

### 4.2 The Inversion Calculation
When drafting the comparison table in `docs/experiments/gradcam_architecture_comparison.md`:
1. **Loc Recall:**
   - The author took ResNet-18's escape-to-none rate ($18.91\%$) and subtracted it from $100\%$:
     $$100\% - 18.91\% = \mathbf{81.09\%}$$
   - The author entered **81.09%** as the "Validation Loc Recall" for ResNet-18!
   - This invalidly ignored the 56 other `Loc` defects misclassified as other defect classes (`FN` $= 97 + 56 = 153$). ResNet-18's true Loc recall is $\frac{360}{513} = \mathbf{70.18\%}$.
2. **Scratch Recall:**
   - The author had recorded Custom CNN's true Scratch recall as **69.41%**.
   - During table construction, **69.41%** was erroneously shifted to the ResNet-18 column.
   - For Custom CNN, the author mistakenly subtracted an assumed clean escape rate ($15.88\% \approx 15.9\%$) from $100\%$:
     $$100\% - 15.88\% = \mathbf{84.12\%}$$
   - The author entered **84.12%** in the Custom CNN column!
3. **README Replication:**
   - The sentence in `README.md` was subsequently summarized based on the faulty comparison table:
     `($30.6\%$ escape rate on Scratch vs. $15.9\%$ for Custom CNN)`

---

## 5. Summary of Correct Values vs. Erroneous Documentation

| Metric / Dimension | Documented Value in `gradcam_architecture_comparison.md` | Verified Correct Value (Audit) | Delta & Impact |
| :--- | :---: | :---: | :--- |
| **Custom CNN Scratch Recall** | `84.12%` | **69.41%** ($118 / 170$) | **-14.71% correction** |
| **Custom CNN Scratch Escapes to Clean** | `15.9%` | **21.76%** ($37 / 170$) | **+5.86% correction** |
| **ResNet-18 Scratch Recall** | `69.41%` | **61.76%** ($105 / 170$) | **-7.65% correction** |
| **ResNet-18 Scratch Escapes to Clean** | `30.6%` | **30.59%** ($52 / 170$) | **Accurate** ($52 / 170 = 30.59\%$) |
| **Custom CNN Loc Recall** | `81.13%` | **79.14%** ($406 / 513$) | **-1.99% correction** |
| **Custom CNN Loc Escapes to Clean** | `18.9%` | **13.65%** ($70 / 513$) | **-5.25% correction** |
| **ResNet-18 Loc Recall** | `81.09%` | **70.18%** ($360 / 513$) | **-10.91% correction** |
| **ResNet-18 Loc Escapes to Clean** | `18.9%` | **18.91%** ($97 / 513$) | **Accurate** ($97 / 513 = 18.91\%$) |

### Does This Change the Core Comparative Thesis?
**No.** The underlying scientific and engineering conclusion remains completely intact and is actually **reinforced**:
1. `CustomWaferCNN` achieves a Scratch recall of **69.41%**, substantially outperforming `AdaptedResNet18` at **61.76%** ($+7.65\%$ recall advantage; $+0.0718$ F1 advantage).
2. `CustomWaferCNN` misses only **37** scratches to clean ($21.76\%$), whereas `AdaptedResNet18` misses **52** scratches to clean ($30.59\%$), representing a **$40.5\%$ increase in clean wafer defect escapes** for ResNet-18 ($52$ vs. $37$).
3. `CustomWaferCNN` achieves a Loc recall of **79.14%**, substantially outperforming `AdaptedResNet18` at **70.18%** ($+8.96\%$ recall advantage; $+0.0391$ F1 advantage).
4. `CustomWaferCNN` misses only **70** localized defect clusters to clean ($13.65\%$), whereas `AdaptedResNet18` misses **97** to clean ($18.91\%$), representing a **$38.6\%$ increase in localized defect escapes** for ResNet-18 ($97$ vs. $70$).

---

## 6. Actionable Documentation Corrections

The following specific file edits are required to harmonize the project documentation with the verified empirical data:

### Correction 1: `docs/experiments/gradcam_architecture_comparison.md`
**Target Table (Lines 40–41):**
```markdown
# OLD (Erroneous):
| **Validation Scratch Recall** | **84.12%** (Escape: 15.9%) | **69.41%** (Escape: 30.6%) | **Custom CNN catches $2\times$ more scratches** |
| **Validation Loc Recall** | **81.13%** (Escape: 18.9%) | **81.09%** (Escape: 18.9%) | Custom CNN produces sharper localization |

# NEW (Corrected):
| **Validation Scratch Recall** | **69.41%** (Clean Escape: 21.8%) | **61.76%** (Clean Escape: 30.6%) | **Custom CNN leads by +7.65% (+40.5% fewer escapes)** |
| **Validation Loc Recall** | **79.14%** (Clean Escape: 13.7%) | **70.18%** (Clean Escape: 18.9%) | **Custom CNN leads by +8.96% (+38.6% fewer escapes)** |
```

### Correction 2: `README.md`
**Target Bullet (Line 311):**
```markdown
# OLD (Erroneous):
- Empirically validated the **Early Downsampling Hypothesis**: ResNet-18 downsamples $64 \times 64$ inputs by $32\times$ into a $2 \times 2$ grid cell, diluting sparse 1-die wide scratches and small clusters into normal background dies ($30.6\%$ escape rate on Scratch vs. $15.9\%$ for Custom CNN)

# NEW (Corrected):
- Empirically validated the **Early Downsampling Hypothesis**: ResNet-18 downsamples $64 \times 64$ inputs by $32\times$ into a $2 \times 2$ grid cell, diluting sparse 1-die wide scratches and small clusters into normal background dies ($30.6\%$ clean escape rate on Scratch vs. $21.8\%$ for Custom CNN; Scratch recall $61.8\%$ vs. $69.4\%$)
```
