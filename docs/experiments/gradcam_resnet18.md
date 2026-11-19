# Phase 6B: Grad-CAM Explainability for AdaptedResNet18 — Experimental Report

**Project:** Semiconductor Wafer Defect Detection & Classification (WM-811K)  
**Model Family:** Adapted ResNet-18 (`AdaptedResNet18`, 11,174,857 parameters)  
**Evaluated Checkpoint:** [`models/checkpoints/resnet18_pretrained_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/resnet18_pretrained_best.pt) (Transfer Learning Best, Validation Macro F1 = **0.8611**)  
**Target Explanation Layer:** `layer4` (resolved to `model.backbone.layer4`)  
**Feature Map Grid:** $2 \times 2 \times 512$ channels ($32\times$ downsampled from $64 \times 64$ input)  
**Input Modality:** Single-channel numerical (`[B, 1, 64, 64]`), Method C letterbox padding  
**Evaluation Partition:** Validation Set ($N_{\text{val}} = 24,705$ wafers, **0% lot leakage**)  
**Held-Out Benchmark:** Test Set ($N_{\text{test}} = 24,705$ wafers, **strictly unvisited**)  
**Benchmark Date:** November 2026  
**Status:** Completed & Synchronized  

---

## 1. Executive Summary & Objective

In Phase 5A, transfer learning using ImageNet pretrained ResNet-18 achieved a Validation Macro F1 of **0.8611**, trailing our lightweight `CustomWaferCNN` baseline (**0.8861**) by **-0.0250 F1 points**. ResNet-18 exhibited notably higher escape rates on thin, localized defect topologies (missed $30.6\%$ of `Scratch` and $18.9\%$ of `Loc` defects).

The objective of Phase 6B is to implement **Gradient-weighted Class Activation Mapping (Grad-CAM)** for `AdaptedResNet18` and systematically investigate:
1. Which spatial regions within the $64 \times 64$ wafer map drive ResNet-18's classification decisions across all 9 classes.
2. The internal attribution patterns behind ResNet-18's missed defects (`Scratch`, `Loc`, `Edge-Loc` falsely classified as `none`).
3. The spatial drivers of false alarms on clean wafers.
4. The structural impact of ResNet-18's $32\times$ downsampling hierarchy ($64 \times 64 \to 2 \times 2$) on spatial attribution fidelity.

---

## 2. Mathematical Formulation & ResNet-18 Hook Architecture

Grad-CAM extracts class-discriminative attributions by backpropagating the score for target class $c$ into the final residual stage of ResNet-18 (`backbone.layer4`).

### 2.1 Layer Resolution & Feature Dimensions
In `AdaptedResNet18`, `layer4` contains two residual BasicBlocks (`BasicBlock(512, 512)`). For an input wafer tensor of shape `[B, 1, 64, 64]`, the downsampling stages proceed as follows:

$$\begin{aligned}
\text{Input} &: [B, 1, 64, 64] \\
\text{conv1 + maxpool} &: [B, 64, 16, 16] \quad (4\times \text{ downsampling}) \\
\text{layer1} &: [B, 64, 16, 16] \\
\text{layer2} &: [B, 128, 8, 8] \quad (8\times \text{ downsampling}) \\
\text{layer3} &: [B, 256, 4, 4] \quad (16\times \text{ downsampling}) \\
\text{layer4} &: [B, 512, 2, 2] \quad (32\times \text{ downsampling})
\end{aligned}$$

At `layer4`, the spatial grid is compressed to only $H=2, W=2$ with $K=512$ feature channels.

### 2.2 Gradient Pooling & Weighted Combination
Gradients of target class score $y^c$ with respect to activation tensor $A^k \in \mathbb{R}^{2 \times 2}$ are globally average pooled across $Z = 2 \times 2 = 4$ locations:

$$\alpha_k^c = \frac{1}{4} \sum_{i=1}^2 \sum_{j=1}^2 \frac{\partial y^c}{\partial A_{i, j}^k}$$

The rectified combination is computed as:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{512} \alpha_k^c A^k \right) \in \mathbb{R}^{2 \times 2}$$

### 2.3 Bilinear Upsampling & Safe Normalization
Because the activation map $L_{\text{Grad-CAM}}^c$ is $2 \times 2$, projecting it back onto the $64 \times 64$ wafer map requires a **$32\times$ spatial magnification** via bilinear interpolation:

$$L_{\text{upsampled}}^c = \text{Interpolate}\left(L_{\text{Grad-CAM}}^c, \text{size}=(64, 64), \text{mode}=\text{"bilinear"}\right)$$

Safe normalization handles low-variance or near-constant maps:

$$\text{CAM}_{\text{norm}} = \begin{cases} 
\frac{L_{\text{upsampled}}^c - \min(L)}{\max(L) - \min(L) + 10^{-8}}, & \text{if } \max(L) - \min(L) > 10^{-8} \\
\mathbf{0}_{64 \times 64}, & \text{otherwise}
\end{cases}$$

---

## 3. Visual Findings Across All 9 Defect Classes

All figures were generated strictly on the validation partition ($N_{\text{val}} = 24,705$ wafers) using [`scripts/generate_resnet18_gradcam.py`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/scripts/generate_resnet18_gradcam.py) and saved in [`data/interim/figures/gradcam/resnet18/`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/).

### 3.1 Correctly Classified Cases Summary

```
                      AdaptedResNet18 (layer4) Spatial Focus
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Class 0: none       --> Flat, uniform field across functional dies          │
  │ Class 1: Center     --> Central hemispherical blob (~30-pixel diameter)     │
  │ Class 2: Donut      --> Diffuse circular ring enclosing the wafer center    │
  │ Class 3: Edge-Loc   --> Peripheral broad quadrant hotspot                   │
  │ Class 4: Edge-Ring  --> Full perimeter border ring attribution             │
  │ Class 5: Loc        --> Broad local lobe spanning 1-2 entire quadrants      │
  │ Class 6: Random     --> Generalized full-wafer diffuse activation           │
  │ Class 7: Scratch    --> Broad diagonal swath across wafer face              │
  │ Class 8: Near-full  --> Saturated wafer-wide heatmap                        │
  └─────────────────────────────────────────────────────────────────────────────┘
```

Representative visual artifacts:
- **Composite 9-Class Overview:** [`resnet18_summary_9classes.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_summary_9classes.png)
- **Class 0 (`none`):** [`resnet18_correct_0_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_0_none.png) (Confidence: 100.0%) — Gradient intensity is low and distributed homogeneously across normal functional dies without focal concentration.
- **Class 1 (`Center`):** [`resnet18_correct_1_center.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_1_center.png) (Confidence: 100.0%) — Correctly highlights wafer center, though the activation contour is broad due to $2 \times 2$ bilinear blurring.
- **Class 2 (`Donut`):** [`resnet18_correct_2_donut.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_2_donut.png) (Confidence: 100.0%) — Annular shape is captured, though the central void is partially encroached by interpolation bleed.
- **Class 3 (`Edge-Loc`):** [`resnet18_correct_3_edge_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_3_edge_loc.png) (Confidence: 100.0%) — Peripheral defect is localized, but the activation covers approximately 25% of the total wafer area.
- **Class 4 (`Edge-Ring`):** [`resnet18_correct_4_edge_ring.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_4_edge_ring.png) (Confidence: 100.0%) — High-intensity perimeter ring attribution tracing edge exclusion zones.
- **Class 5 (`Loc`):** [`resnet18_correct_5_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_5_loc.png) (Confidence: 100.0%) — Local defect cluster activates the correct quadrant.
- **Class 6 (`Random`):** [`resnet18_correct_6_random.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_6_random.png) (Confidence: 99.8%) — Diffuse full-wafer excitation reflecting scattered Poisson defect distribution.
- **Class 7 (`Scratch`):** [`resnet18_correct_7_scratch.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_7_scratch.png) (Confidence: 99.8%) — Wide diagonal band following the scratch orientation.
- **Class 8 (`Near-full`):** [`resnet18_correct_8_near_full.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_correct_8_near_full.png) (Confidence: 100.0%) — Complete spatial saturation.

---

## 4. Root-Cause Analysis of Missed Defects

During Phase 5A validation, ResNet-18 missed 52 `Scratch` samples ($30.6\%$ escape rate) and 97 `Loc` samples ($18.9\%$ escape rate). Grad-CAM illuminates why:

### 4.1 Missed `Scratch` (True Scratch $\to$ Predicted `none`)
- **Attribution to Predicted Class (`none`):** [`resnet18_missed_scratch_pred_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_missed_scratch_pred_none.png)
  - ResNet-18 focuses globally on the large background of normal functional dies ($98\%+$ of wafer area).
  - Because each $2 \times 2$ grid cell aggregates $32 \times 32 = 1,024$ input pixels, a narrow 1-die wide scratch track ($10\text{–}15$ defective pixels) contributes only $\sim 1\text{–}1.5\%$ of the total signal within a cell.
  - The dominant functional die signal completely swamps the sparse defect gradient, leading the model to confidently output `none` ($98.6\%$ confidence).
- **Attribution to True Class (`Scratch`):** [`resnet18_missed_scratch_target_true_class.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_missed_scratch_target_true_class.png)
  - When forcing backpropagation with respect to the `Scratch` logit, weak activation appears along the scratch axis, but the signal magnitude is insufficient to overcome the `none` prior.

### 4.2 Missed `Loc` (True Loc $\to$ Predicted `none`)
- **Attribution to Predicted Class (`none`):** [`resnet18_missed_loc_pred_none.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_missed_loc_pred_none.png)
  - A small localized defect cluster near the wafer periphery fails to trigger a decisive response in `layer4`.
  - The $2 \times 2$ spatial pooling causes the cluster to be spatially averaged with surrounding normal dies, yielding a low-magnitude activation that is classified as background noise.
- **Attribution to True Class (`Loc`):** [`resnet18_missed_loc_target_true_class.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_missed_loc_target_true_class.png)

---

## 5. False Alarms on Clean Wafers

- **Clean Wafer Predicted as `Center`:** [`resnet18_false_alarm_1_center.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_false_alarm_1_center.png)
  - On a clean wafer with slight random surface noise near the center, ResNet-18 over-attends to the central coordinates.
  - ImageNet-derived inductive biases, which favor center-focused subject detection, cause the model to mistake benign noise for a center defect cluster.
- **Clean Wafer Predicted as `Loc`:** [`resnet18_false_alarm_2_loc.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/data/interim/figures/gradcam/resnet18/resnet18_false_alarm_2_loc.png)

---

## 6. Engineering Limitations & Disclaimers

> [!IMPORTANT]
> **Spatial Coarseness Disclaimer:** Due to ResNet-18's $32\times$ downsampling hierarchy ($64 \times 64 \to 2 \times 2$), heatmaps generated from `layer4` are inherently coarse and diffuse. A single $2 \times 2$ activation grid cell covers an entire quadrant ($32 \times 32$ dies) of the input wafer.

> [!WARNING]
> **Interpretability Bounds:** Grad-CAM provides post-hoc visual attribution of class-discriminative evidence; it **does not** constitute exact die-level segmentation, defect boundary delineation, or proof of causal physical mechanism.
