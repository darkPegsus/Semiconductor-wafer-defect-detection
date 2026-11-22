# WaferVision AI: User Manual & Application Deployment Guide

**Application:** WaferVision AI — Professional Semiconductor Wafer Defect Inspection & Explainability Tool  
**Theme:** Light Industrial Fab Interface (High-Contrast, Clean Neutral Palette)  
**Engine:** `CustomWaferCNN` (520,681 parameters, single-channel 64×64 input)  
**Sealed Checkpoint:** [`models/checkpoints/exp1_custom_cnn_baseline_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/exp1_custom_cnn_baseline_best.pt)  
**Checkpoint SHA-256:** `b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8`  
**Held-Out Test Performance:** **97.62% Accuracy**, **0.8746 Macro F1**, **0.95% False Alarm Rate**  
**Default Launch Port:** `http://localhost:8501`  
**Status:** Operational Production Deployment  

---

## 1. System Architecture & Tech Stack

WaferVision AI is an industrial-grade inspection application designed for semiconductor yield learning and fabrication quality control. It packages the benchmark-winning `CustomWaferCNN` model into a clean, operational inspection tool.

```
                     WaferVision AI Application Architecture
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                        Streamlit Frontend UI (Light Industrial)             │
  │   - Wafer Inspection (Primary View)    - Model Performance (Benchmark)      │
  │   - 3-Panel Grad-CAM Explainability    - About (Architecture & Boundaries)  │
  └───────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           Application Services Layer                        │
  │  ┌─────────────────────────┐ ┌───────────────────────┐ ┌──────────────────┐  │
  │  │ Preprocessing Service   │ │   Inference Service   │ │   Explainability │  │
  │  │ - Input Standardization │ │ - Frozen Checkpoint   │ │ - Grad-CAM Hook  │  │
  │  │ - Method C Letterbox    │ │ - Softmax Calibration │ │ - conv_block4    │  │
  │  │ - Tensor [1, 1, 64, 64] │ │ - Latency Benchmarking│ │ - Die Overlay    │  │
  │  └─────────────────────────┘ └───────────────────────┘ └──────────────────┘  │
  └───────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                       Deep Learning Backend (PyTorch)                       │
  │   - CustomWaferCNN Model (520,681 Frozen Parameters, 0 Trainable)           │
  │   - SHA-256 Verified Checkpoint Integrity Check                             │
  │   - Automatic Device Resolution (CUDA Acceleration or CPU Execution)        │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Light Industrial UI & Accessibility Design

The user interface follows strict semiconductor fabrication tool design standards:

- **Background:** Clean neutral light grey (`#f8fafc`).
- **Panels & Cards:** Pure white (`#ffffff`) with subtle structural borders (`#e2e8f0`).
- **Primary Typography:** Deep charcoal / dark navy (`#0f172a`) ensuring maximum readability and accessibility.
- **Secondary Text:** Clear slate grey (`#475569`).
- **Defect Indicators:** High-contrast crimson / red badge (`#fee2e2` bg, `#991b1b` text, `#fca5a5` border).
- **Clean/Normal Indicators:** High-contrast emerald badge (`#f0fdf4` bg, `#166534` text, `#86efac` border).
- **Theme Enforcement:** `.streamlit/config.toml` sets `base = "light"`, avoiding conflicts with host OS dark modes.

---

## 3. Operational Navigation & Page Structure

The application provides three dedicated sections accessible via the sidebar:

1. **`🔍 Wafer Inspection` (Default Operational View):**
   - Compact status header (Model, Checkpoint verification, Device).
   - Dual-mode wafer input (Upload vs. Demonstration Samples).
   - Wafer die preview and active morphology statistics.
   - Prominent **`ANALYZE WAFER`** action button.
   - Classification card, softmax confidence, latency, and full 9-class probability bar chart.
   - Neutral **Inspection Notes & Engineering Guidance** card.
   - Dedicated **Model Explainability | Grad-CAM** 3-panel attribution viewer.

2. **`📊 Model Performance` (Historical Benchmark Evaluation):**
   - Sealed held-out test evaluation results on WM-811K ($N = 24,705$ wafers, $1,537$ lots).
   - Key KPI cards: Test Accuracy ($97.62\%$), Macro F1 ($0.8746$), Precision ($89.09\%$), Recall ($86.22\%$), Weighted F1 ($0.9761$).
   - Complete per-class breakdown table and confusion matrix heatmap.

3. **`ℹ️ About` (Technical Reference & Operational Boundaries):**
   - Technical summary of `CustomWaferCNN` (4 conv blocks, Mish activations, CBAM attention, $4 \times 4$ spatial feature retention).
   - WM-811K benchmark dataset specifications.
   - Visible model limitations (Scratch recall: $64.12\%$, Loc recall: $77.97\%$, coarse Grad-CAM resolution).
   - Active checkpoint path and SHA-256 verification hash.

---

## 4. How to Launch the Application

From the project root:

```powershell
# Direct launch via Streamlit (Recommended)
.\.venv\Scripts\streamlit run app/main.py

# Or via dedicated runner script
.\.venv\Scripts\python scripts/run_app.py
```

Access the interface in your browser at: `http://localhost:8501`.

---

## 5. Visual Inspection Screenshots

Captured at standard laptop viewport ($1366 \times 768$):

| View | Screenshot File | Description |
| :--- | :--- | :--- |
| **Inspection (Pre-Prediction)** | [`screenshot_1_inspection_before_prediction.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/deployment/screenshots/screenshot_1_inspection_before_prediction.png) | Compact header, sample selection, wafer preview, die stats, and `ANALYZE WAFER` button. |
| **Inspection (Post-Prediction)** | [`screenshot_2_inspection_after_prediction.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/deployment/screenshots/screenshot_2_inspection_after_prediction.png) | Classification status card, confidence, latency, 9-class probability distribution, and inspection notes. |
| **Grad-CAM Explainability** | [`screenshot_3_gradcam_visualization.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/deployment/screenshots/screenshot_3_gradcam_visualization.png) | 3-Panel visualization (Original wafer, Grad-CAM heatmap, die-preserving overlay). |
| **Model Performance Dashboard** | [`screenshot_4_model_performance.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/deployment/screenshots/screenshot_4_model_performance.png) | Sealed test benchmark KPIs, per-class metrics table, and confusion matrix. |
| **Technical About Reference** | [`screenshot_5_about_page.png`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/docs/deployment/screenshots/screenshot_5_about_page.png) | Architecture overview, WM-811K specifications, operational limitations, and checkpoint integrity. |

---

## 6. End-to-End Inspection Pipeline

1. **Input Normalization:** Uploaded files or demonstration samples are mapped to ternary arrays $\{0: \text{exterior}, 1: \text{normal die}, 2: \text{defective die}\}$.
2. **Method C Letterbox Padding:** Scaled with aspect-ratio preservation to $64 \times 64$ with symmetric zero padding.
3. **Inference Execution:** Evaluated under `torch.inference_mode()` with latency measurement.
4. **Probability Calibration:** Computes softmax probabilities over all 9 canonical defect classes.
5. **Neutral Inspection Notes:** Analyzes confidence and secondary class margin to provide engineering guidance without fabricating unsupported equipment root causes.
6. **Grad-CAM Feature Attribution:** Hooks into `features.conv_block4` to generate spatial attribution heatmaps.

---

## 7. Model Limitations & Human Review Protocol

> [!IMPORTANT]
> **Operational Boundaries:**
> 1. **Engineering Review Guideline:** Predictions with softmax confidence below $85.0\%$ or small class margin are flagged for secondary human verification.
> 2. **Sensitivity Boundaries:** Faint mechanical scratches exhibit a test recall of $64.12\%$ ($22.94\%$ escape rate to clean), while localized clusters exhibit $77.97\%$ recall ($11.89\%$ clean escape rate). Visual verification is recommended.
> 3. **Explainability Bounds:** Grad-CAM provides coarse attribution from the $8 \times 8$ feature map; it does not produce micron-accurate die segmentation masks.
> 4. **Decision Scope:** WaferVision AI performs morphological pattern recognition and does not diagnose specific tool hardware malfunctions or automatically scrap wafers.
