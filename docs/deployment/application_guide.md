# WaferVision AI: User Manual & Application Deployment Guide

**Application:** WaferVision AI — Interactive Semiconductor Wafer Defect Detection & Explainability Dashboard  
**Engine:** `CustomWaferCNN` (520,681 parameters, single-channel 64×64 input)  
**Sealed Checkpoint:** [`models/checkpoints/exp1_custom_cnn_baseline_best.pt`](file:///c:/Users/Aditya%20indalkar/Projects/Semiconductor-wafer-defect-detection/models/checkpoints/exp1_custom_cnn_baseline_best.pt)  
**Checkpoint SHA-256:** `b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8`  
**Held-Out Test Performance:** **97.62% Accuracy**, **0.8746 Macro F1**, **0.95% False Alarm Rate**  
**Default Launch Port:** `http://localhost:8501`  
**Status:** Production Ready  

---

## 1. System Architecture & Overview

WaferVision AI is an industrial-grade web application designed for semiconductor yield learning and fabrication quality control. It operationalizes our benchmark-winning, ultra-lightweight `CustomWaferCNN` model into an interactive graphical interface.

```
                     WaferVision AI Application Architecture
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                            Streamlit Frontend UI                            │
  │   - Wafer Analysis (Live Inference)   - Benchmark Performance Dashboard     │
  │   - Interactive Grad-CAM Viewer       - Architecture & Fab Guidelines       │
  └───────────────────────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           Application Services Layer                        │
  │  ┌─────────────────────────┐ ┌───────────────────────┐ ┌──────────────────┐  │
  │  │ Preprocessing Service   │ │   Inference Service   │ │   Explainability │  │
  │  │ - Format Validation     │ │ - Frozen Checkpoint   │ │ - Grad-CAM Hook  │  │
  │  │ - Method C Letterbox    │ │ - Softmax Probabilities│ │ - conv_block4    │  │
  │  │ - Tensor [1, 1, 64, 64] │ │ - Latency Measurement │ │ - Die Overlay    │  │
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

## 2. Environment Setup & Dependencies

The application runs locally on Windows, macOS, or Linux using Python 3.12.

### Prerequisites
- Python 3.12 or compatible environment
- PyTorch 2.6.0+ (CUDA optional for GPU acceleration)
- Streamlit $\ge$ 1.30.0
- Plotly $\ge$ 5.18.0
- OpenCV-Python, Matplotlib, NumPy, Pillow, Loguru

### Installation
From the project root:

```bash
# Activate your existing virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or install dependencies
pip install -r requirements.txt
```

---

## 3. How to Launch the Application

You can launch the web application using either of two methods:

### Method A: Streamlit Direct Launch (Recommended)
```bash
streamlit run app/main.py
```

### Method B: Convenience Python Runner
```bash
python scripts/run_app.py
```

Upon launch, the server will output:
```text
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser.

---

## 4. Input Modes & Supported Formats

WaferVision AI provides two flexible modes for defect classification:

### Mode 1: Demonstration Sample Wafers (1-Click Instant Demo)
Pre-packaged with 9 representative validation wafer samples from the WM-811K benchmark (one for each canonical class):
- `none` (Normal functional wafer)
- `Center` (CMP slurry / CVD gas center cluster)
- `Donut` (Toroidal thermal anneal gradient)
- `Edge-Loc` (Edge bead removal / clamp stress)
- `Edge-Ring` (Plasma sheath perimeter drift)
- `Loc` (Localized particulate splatter)
- `Random` (Poisson microparticulates)
- `Scratch` (Mechanical robot slippage / CMP pad grit)
- `Near-full` (Catastrophic tool failure)

Users can test any defect topology instantly without uploading files.

### Mode 2: Custom Wafer Map Upload
Supports user-uploaded files:
- **Image Formats:** `.png`, `.jpg`, `.jpeg`, `.bmp` (RGB or Grayscale).
- **NumPy Matrix:** `.npy` (2D discrete integer or float array).

#### Validation Rules:
- **Minimum Grid Size:** $8 \times 8$ dies.
- **Maximum Grid Size:** $2048 \times 2048$ pixels.
- **Value Semantics:**
  - `0`: Exterior space / canvas padding
  - `1`: Normal functional silicon die
  - `2`: Defective / failed silicon die

---

## 5. End-to-End Inference Workflow

When a wafer map is submitted:

1. **Input Normalization & Parsing:** Grayscale/RGB pixel intensities are mapped to discrete ternary die states $\{0, 1, 2\}$.
2. **Method C Letterbox Padding:** The wafer map is proportionally scaled using nearest-neighbor interpolation to fit within a $64 \times 64$ canvas, preserving the physical die aspect ratio. The remaining margins are symmetrically padded with exterior zero-fill.
3. **Single-Channel Tensor Construction:** The $64 \times 64$ map is cast to a single-channel float tensor `[1, 1, 64, 64]`.
4. **Deterministic Inference:** The frozen `CustomWaferCNN` executes forward evaluation with `torch.inference_mode()`.
5. **Diagnostics & Probability Distribution:** Softmax probabilities are calculated for all 9 classes, and inference latency is recorded.
6. **Grad-CAM Explainability Generation:** Forward and backward hooks extract activation feature maps and gradients at `features.conv_block4`, generating a die-preserving heatmap overlay.

---

## 6. Interpretability & Grad-CAM Controls

The **Grad-CAM Spatial Attribution** panel allows interactive engineering inspection:
- **Target Class Selector:** Defaults to the top-predicted class. Users can select any alternative defect class to inspect **counterfactual attributions** (i.e. what spatial features would support classifying this wafer as `Loc` or `Scratch`?).
- **Alpha Blending Slider:** Dynamically adjusts heatmap transparency ($0.2\text{–}0.8$) over the silicon dies.
- **Die-Preserving Overlay:** Functional dies appear in slate gray, while defective dies are superimposed in vivid crimson on top of the jet heatmap, ensuring physical die boundaries are never obscured.

---

## 7. Model Limitations & Human Review Protocol

> [!IMPORTANT]
> **Operational Triage Guidelines:**
> 1. **Informational Review Flag:** The application displays an automated warning if prediction confidence is below $85\%$ or if a secondary class exhibits high probability ($<25\%$ gap). In high-volume manufacturing, flagged wafers should receive manual review.
> 2. **Sparse Defect Recall Boundaries:** On the held-out test set, `Scratch` recall is **64.12%** ($22.94\%$ escape rate to clean) and `Loc` recall is **77.97%** ($11.89\%$ escape rate to clean). Fine, single-die scratch tracks require careful visual verification.
> 3. **Not a Segmentation Tool:** Grad-CAM reflects class-discriminative feature weighting, not micron-accurate die segmentation masks or physical causal failure proofs.
> 4. **Fab Node Transferability:** Qualified strictly on WM-811K benchmark wafers. Novel fabrication tools or proprietary wafer grid geometries require domain-specific calibration.

---

## 8. Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **`FileNotFoundError: Checkpoint not found`** | Checkpoint missing from `models/checkpoints/` | Ensure `exp1_custom_cnn_baseline_best.pt` is present. |
| **`ValueError: Checkpoint integrity check failed`** | Checkpoint corrupted or modified | Re-verify SHA-256 hash matches `b9df80c522dd...`. |
| **`ImportError: No module named 'src'`** | Python path not resolving project root | Run from project root or use `python scripts/run_app.py`. |
| **`CUDA Out of Memory`** | VRAM constrained by another process | Application automatically falls back to CPU, requiring under 150 MB. |
| **Streamlit Port Collision (`Port 8501 is already in use`)** | Another Streamlit process active | Run with custom port: `streamlit run app/main.py --server.port 8502`. |
