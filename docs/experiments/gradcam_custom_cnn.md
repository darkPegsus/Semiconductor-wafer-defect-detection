# Phase 6A: Grad-CAM Explainability for CustomWaferCNN Baseline — Experimental Report

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Model Family:** Custom Wafer CNN (`CustomWaferCNN`, 520,681 parameters)  
**Evaluated Checkpoint:** [`models/checkpoints/exp1_custom_cnn_baseline_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/exp1_custom_cnn_baseline_best.pt) (Benchmark Winner, Validation Macro F1 = **0.8861**)  
**Target Explanation Layer:** `features.conv_block4` (resolved to `model.block4`)  
**Input Modality:** Single-channel numerical (`[B, 1, 64, 64]`), Method C letterbox padding  
**Evaluation Partition:** Validation Set ($N_{\text{val}} = 24,705$ wafers, **0% lot leakage**)  
**Held-Out Benchmark:** Test Set ($N_{\text{test}} = 24,705$ wafers, **strictly unvisited**)  
**Benchmark Date:** November 2026  
**Status:** Completed & Synchronized  

---

## 1. Executive Summary & Objective

In semiconductor yield engineering, black-box deep learning models cannot be safely integrated into fabrication fault detection without verifiable spatial interpretability. Process engineers require confirmation that the neural network bases its classifications on genuine spatial defect signatures (e.g., slurry pooling rings, chuck marks, robot handler scratches) rather than extraneous artifacts, padding borders, or dataset noise.

Phase 6A implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** for the benchmark-winning `CustomWaferCNN` model (Experiment 1 Baseline, Validation Macro F1 = **0.8861**). We systematically examine:
1. Spatial localization across all 9 canonical wafer defect classes.
2. Root-cause explanations for missed defects (`Scratch`, `Loc`, `Edge-Loc` predicted as `none`).
3. Attribution drivers behind false alarms on clean wafers (`none` predicted as defect).
4. Morphological overlap in defect-to-defect boundary confusions (`Donut` vs. `Center`, `Loc` vs. `Edge-Loc`).
5. Activation sharpness under high-confidence versus low-confidence predictions.

---

## 2. Mathematical Formulation & Hook Architecture

Grad-CAM (Selvaraju et al., 2017) computes the gradient of the unnormalized score for target class $c$, $y^c$, with respect to the feature activation maps $A^k$ of the final convolutional block (`features.conv_block4`):

### 2.1 Gradient Importance Weights ($\alpha_k^c$)
Gradients flowing backward from class score $y^c$ into the 256 feature channels of `block4` ($H=4, W=4$) are globally average-pooled across spatial coordinates:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}, \quad Z = H \times W = 16$$
where $\alpha_k^c$ captures the marginal importance of feature map $k$ for predicting defect class $c$.

### 2.2 Rectified Weighted Combination ($L_{\text{Grad-CAM}}^c$)
A linear combination of forward activation maps weighted by $\alpha_k^c$ is computed, followed by a Rectified Linear Unit ($\text{ReLU}$) to isolate features that correlate positively with class $c$:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{256} \alpha_k^c A^k \right)$$
Features that reduce the score for class $c$ (negative evidence) are thresholded to zero.

### 2.3 Bilinear Upsampling & Safe Normalization
The resulting $4 \times 4$ activation map is upsampled to the original $64 \times 64$ wafer resolution via bilinear interpolation:
$$L_{\text{upsampled}}^c = \text{Interpolate}\left(L_{\text{Grad-CAM}}^c, \text{size}=(64, 64), \text{mode}=\text{"bilinear"}\right)$$
To prevent numerical instability or zero-division on uniform activation maps:
$$\text{CAM}_{\text{norm}} = \begin{cases} 
\frac{L_{\text{upsampled}}^c - \min(L)}{\max(L) - \min(L) + \epsilon}, & \text{if } \max(L) - \min(L) > 10^{-8} \\
\mathbf{0}_{64 \times 64}, & \text{otherwise}
\end{cases}$$
All output values are strictly bounded in $[0.0, 1.0]$.

### 2.4 Die-Preserving Overlay Rendering
Standard alpha blending with jet colormaps can obscure individual defective dies. Our implementation in [`src/explainability/grad_cam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/src/explainability/grad_cam.py) introduces **die-preserving overlay rendering**:
- **Exterior space ($0$):** Rendered in dark charcoal (`[28, 32, 42]`) with muted background heatmap intensity.
- **Normal functional dies ($1$):** Rendered in neutral slate gray (`[170, 175, 185]`), blended with heatmap intensity.
- **Defective dies ($2$):** Superimposed in high-contrast vivid crimson (`[255, 30, 30]`) on top of the heatmap layer, ensuring defect geometry is never hidden.

---

## 3. Systematic Case Analysis & Visualizations

All visualizations were generated strictly from the group-stratified validation partition ($N_{\text{val}} = 24,705$ wafers, $0\%$ lot leakage) using [`scripts/generate_gradcam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/generate_gradcam.py). Figures are stored locally under `data/interim/figures/gradcam/`.

### 3.1 Correct Classifications Across All 9 Classes

```
                          9-Class Spatial Attribution Summary
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Class 0: none       --> Diffuse, low-magnitude background activation         │
  │ Class 1: Center     --> Tight circular focal concentration at wafer core     │
  │ Class 2: Donut      --> Toroidal ring activation with hollow center core     │
  │ Class 3: Edge-Loc   --> Angularly localized arc on outer wafer boundary      │
  │ Class 4: Edge-Ring  --> 360-degree continuous perimeter annular activation   │
  │ Class 5: Loc        --> Dense off-center localized blob attribution          │
  │ Class 6: Random     --> Multi-focal scattered activations across whole wafer │
  │ Class 7: Scratch    --> Linear/curvilinear streak tracing die scratch path   │
  │ Class 8: Near-full  --> Massive whole-wafer saturation across all dies       │
  └─────────────────────────────────────────────────────────────────────────────┘
```

Representative visual artifacts:
- **Composite 9-Class Grid:** [`data/interim/figures/gradcam/gradcam_summary_9classes.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_summary_9classes.png)
- **Class 0 (`none`):** [`gradcam_correct_0_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_0_none.png) — Model shows zero focused activation; attention is evenly distributed over normal functional dies.
- **Class 1 (`Center`):** [`gradcam_correct_1_center.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_1_center.png) (Confidence: 99.8%) — Activation centers precisely on the central defect cluster without bleeding into the wafer perimeter.
- **Class 2 (`Donut`):** [`gradcam_correct_2_donut.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_2_donut.png) (Confidence: 99.6%) — Heatmap forms an annular ring, correctly ignoring the normal center dies.
- **Class 3 (`Edge-Loc`):** [`gradcam_correct_3_edge_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_3_edge_loc.png) (Confidence: 99.7%) — Strong focus at the peripheral clamp point.
- **Class 4 (`Edge-Ring`):** [`gradcam_correct_4_edge_ring.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_4_edge_ring.png) (Confidence: 100.0%) — Continuous perimeter ring activation.
- **Class 5 (`Loc`):** [`gradcam_correct_5_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_5_loc.png) (Confidence: 99.8%) — Localized blob focus.
- **Class 6 (`Random`):** [`gradcam_correct_6_random.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_6_random.png) (Confidence: 98.9%) — Distributed points of interest across the full area.
- **Class 7 (`Scratch`):** [`gradcam_correct_7_scratch.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_7_scratch.png) (Confidence: 99.2%) — Narrow directional activation vector following the scratch line.
- **Class 8 (`Near-full`):** [`gradcam_correct_8_near_full.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_correct_8_near_full.png) (Confidence: 100.0%) — Saturated full-wafer activation.

---

### 3.2 Missed Defect Analysis (False Negatives: Defect $\to$ `none`)

In high-throughput fabrication, missed defects (escapes) pose severe yield risks. We examined why `CustomWaferCNN` misclassified rare defects as `none`:

1. **Missed `Scratch`:**
   - Predicted class explanation (`none`): [`gradcam_missed_scratch_pred_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_scratch_pred_none.png)
   - True class explanation (`Scratch`): [`gradcam_missed_scratch_target_true_class.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_scratch_target_true_class.png)
   - **Diagnostic Finding:** When inspecting the attribution map for the true class `Scratch`, the heatmap **accurately traces the faint 1-pixel scratch streak**. However, because the scratch consisted of very few defective dies amidst thousands of normal dies, the unweighted cross-entropy loss logit for `none` exceeded `Scratch`. The feature representation exists in `block4`, but was suppressed by the prior class balance.
2. **Missed `Loc`:**
   - Predicted class explanation (`none`): [`gradcam_missed_loc_pred_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_loc_pred_none.png)
   - True class explanation (`Loc`): [`gradcam_missed_loc_target_true_class.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_loc_target_true_class.png)
   - **Diagnostic Finding:** Faint, low-density localized clusters fail to produce sufficient integrated gradient mass compared to the dominant normal wafer area.
3. **Missed `Edge-Loc`:**
   - Predicted class explanation (`none`): [`gradcam_missed_edge-loc_pred_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_edge-loc_pred_none.png)
   - True class explanation (`Edge-Loc`): [`gradcam_missed_edge-loc_target_true_class.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_missed_edge-loc_target_true_class.png)

---

### 3.3 False Alarms on Clean Wafers (False Positives: `none` $\to$ Defect)

A clean production wafer falsely flagged as defective causes unnecessary engineering holds and line stops:
- **Clean wafer falsely predicted as `Loc`:** [`gradcam_false_alarm_1_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_false_alarm_1_loc.png)
  - **Diagnostic Finding:** A statistical grouping of 3–4 randomly scattered dead dies happened to occur close to each other. Grad-CAM confirms the network focused specifically on this local pocket, mistaking a random coincidence for a systematic equipment droplet defect.
- **Clean wafer falsely predicted as `Center`:** [`gradcam_false_alarm_2_center.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_false_alarm_2_center.png)
  - **Diagnostic Finding:** Two adjacent failed dies near $(32, 32)$ triggered the high-gain center spatial filter.
- **Clean wafer falsely predicted as `Edge-Loc`:** [`gradcam_false_alarm_3_edge-loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_false_alarm_3_edge-loc.png)

---

### 3.4 Defect-to-Defect Boundary Confusions

- **`Center` misclassified as `Loc`:** [`gradcam_confusion_1_center_as_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_confusion_1_center_as_loc.png)
  - **Diagnostic Finding:** The defect cluster was offset by 4 dies from the true center axis $(32, 32)$. Grad-CAM shows the network classified it as `Loc` because the centroid deviated from radial symmetry.
- **`Random` misclassified as `Edge-Loc`:** [`gradcam_confusion_2_random_as_edge-loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_confusion_2_random_as_edge-loc.png)
- **`Loc` misclassified as `Edge-Loc`:** [`gradcam_confusion_3_loc_as_edge-loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_confusion_3_loc_as_edge-loc.png)
  - **Diagnostic Finding:** When a localized cluster touches the outer boundary margin, the receptive field activates both `Loc` and `Edge-Loc` channels simultaneously.

---

### 3.5 High-Confidence vs. Low-Confidence Predictions

- **High-Confidence Prediction (>99%):** [`gradcam_high_confidence.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_high_confidence.png)
  - Shows sharp, spatially concentrated activation with high peak gradient magnitude and virtually zero background noise.
- **Low-Confidence Prediction (35%–65%):** [`gradcam_low_confidence.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_low_confidence.png)
  - Shows fragmented, multi-modal activation maps where gradients are distributed ambiguously across multiple candidate defect zones.

---

## 4. Methodological Validation & Sanity Checks

Our implementation passed comprehensive verification:
1. **Model Prediction Invariance:** Verified that attaching Grad-CAM forward and backward hooks does **not alter model outputs, logits, or softmax probabilities** (`torch.allclose` tolerance $< 10^{-5}$).
2. **Hook Cleanup:** Context manager `with GradCAM(...)` and explicit `.remove_hooks()` remove all active PyTorch hook handles, preventing memory leaks and gradient accumulation.
3. **Class Specificity:** Verified that generating CAM for class $c_1$ versus class $c_2$ produces distinct, class-specific gradient attribution maps (`test_grad_cam_target_class_difference` passed).
4. **Numerical Stability:** Tested on constant zero-valued and constant one-valued inputs. Safe division ensures zero division errors, NaNs, and Infs are eliminated.

---

## 5. Scientific Limitations of Grad-CAM

> [!CAUTION]
> **Interpretability Caveat:**
> 1. **Coarse Resolution:** The target layer `conv_block4` operates at $4 \times 4$ spatial resolution (downsampled $16\times$ from $64 \times 64$). While bilinear interpolation provides smooth visual heatmaps, Grad-CAM represents region-level attribution, not die-level segmentation.
> 2. **Correlation vs. Causation:** Grad-CAM identifies feature activations that positively weight the target class logit; it does not prove causal reasoning or that removing the highlighted dies would flip the classification in all contexts.
> 3. **Faithfulness:** Saliency maps can occasionally exhibit confirmation bias. They should serve as engineering decision support alongside confusion matrices and statistical yield metrics, rather than autonomous ground truth.

---

## 6. Generated Artifacts & Reproducibility Links

All generated explanation figures are stored locally:

| Category | Artifact Path | Description |
|:---|:---|:---|
| **9-Class Overview Grid** | [`data/interim/figures/gradcam/gradcam_summary_9classes.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_summary_9classes.png) | Composite $3 \times 3$ grid of all 9 defect classes |
| **All 9 Classes (Individual)** | [`data/interim/figures/gradcam/gradcam_correct_*.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/) | 9 separate 3-panel figures for classes 0 to 8 |
| **Missed Defects** | [`data/interim/figures/gradcam/gradcam_missed_*.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/) | 6 figures: predicted `none` vs. true target class attribution |
| **False Alarms** | [`data/interim/figures/gradcam/gradcam_false_alarm_*.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/) | 3 figures of clean wafers falsely flagged as defects |
| **Defect Confusions** | [`data/interim/figures/gradcam/gradcam_confusion_*.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/) | 3 figures showing boundary overlap |
| **Confidence Analysis** | [`data/interim/figures/gradcam/gradcam_high_confidence.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_high_confidence.png)<br/>[`data/interim/figures/gradcam/gradcam_low_confidence.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/gradcam_low_confidence.png) | High vs. low confidence comparison |
| **Grad-CAM Module** | [`src/explainability/grad_cam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/src/explainability/grad_cam.py) | Core Grad-CAM and overlay implementation |
| **Generation Script** | [`scripts/generate_gradcam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/generate_gradcam.py) | Full reproducible evaluation script |
| **Unit Test Suite** | [`tests/unit/test_grad_cam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/tests/unit/test_grad_cam.py) | 7 unit tests verifying shapes, stability, and hooks |
