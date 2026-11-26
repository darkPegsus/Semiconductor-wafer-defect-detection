# Streamlit Community Cloud Deployment Guide: WaferVision AI

This guide documents the architecture, configuration, and step-by-step procedures to deploy **WaferVision AI** to **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io)).

---

## 1. Deployment Specifications & Environment

| Parameter | Value |
| :--- | :--- |
| **Repository** | `https://github.com/darkPegsus/Semiconductor-wafer-defect-detection` |
| **Target Branch** | `main` |
| **Application Entrypoint** | `app/main.py` |
| **Target Python Runtime** | `3.12` |
| **Compute Architecture** | CPU-Only (x86_64 Debian/Ubuntu container) |
| **Memory Allocation** | ~1.0 GB RAM (App steady-state footprint: ~250–350 MB) |
| **Dependencies Manifest** | `requirements.txt` |
| **UI Theme & Config** | `.streamlit/config.toml` |

---

## 2. Checkpoint Delivery Architecture (Option A: Public GitHub Release Asset)

### A. The Challenge
The trained deep learning model weights (`exp1_custom_cnn_baseline_best.pt`) are 6.28 MB and excluded from Git by `.gitignore` to prevent binary repository bloat. On a fresh clone in Streamlit Community Cloud, local checkpoint files do not exist.

### B. The Automated Delivery Mechanism
WaferVision AI uses an automated, cryptographic checkpoint retrieval service (`app/utils/checkpoint_manager.py`):

1. **Existence Check:** On application startup (`run_startup_healthcheck`), the app checks if `models/checkpoints/exp1_custom_cnn_baseline_best.pt` exists.
2. **Cold-Start Download:** If absent, it streams the asset from the public GitHub Release asset URL:
   ```
   https://github.com/darkPegsus/Semiconductor-wafer-defect-detection/releases/download/v1.0.0/exp1_custom_cnn_baseline_best.pt
   ```
   *(Overrideable at runtime via the `WAFER_CHECKPOINT_URL` environment variable).*
3. **Atomic Staging & Cryptographic Verification:**
   - The file is streamed to a temporary file (`exp1_custom_cnn_baseline_best.pt.tmp.<uuid>`) in the same directory.
   - The SHA-256 digest is computed on the fly.
   - The computed digest is compared against the frozen benchmark hash:
     ```
     b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8
     ```
   - If the hash matches, `os.replace` atomically moves the temporary file to its final cache location.
   - If corrupted or incomplete, the temporary file is immediately unlinked (deleted) and an error is raised. Corrupted weights are **never** loaded.
4. **Subsequent Cached Startup:** The model is cached on the local container disk and in memory via `@st.cache_resource`, ensuring subsequent reruns and user requests incur zero download overhead.

---

## 3. Runtime Dependencies & CPU Optimization

To ensure fast deployment builds and stay well within memory limits, `requirements.txt` includes:
```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.4.0
torchvision>=0.19.0
```
This instructs `pip` on Linux containers to select lightweight CPU-only PyTorch wheels (~160 MB) instead of standard CUDA wheels (>2.5 GB), reducing container provisioning time from minutes to seconds.

Headless Computer Vision is guaranteed via `opencv-python-headless>=4.10.0.84`, eliminating any requirement for native display libraries (`libGL.so.1`, Qt, X11).

---

## 4. Pre-Deployment Step: Publish GitHub Release v1.0.0

Before deploying on Streamlit Cloud, the public release asset must be available on GitHub so the application can download it anonymously:

1. **Verify Local Checkpoint Integrity:**
   Ensure the local file matches the frozen hash:
   ```bash
   python -c "import hashlib; assert hashlib.sha256(open('models/checkpoints/exp1_custom_cnn_baseline_best.pt', 'rb').read()).hexdigest() == 'b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8', 'Hash mismatch'"
   ```
2. **Tag Release v1.0.0:**
   ```bash
   git tag -a v1.0.0 -m "WaferVision AI v1.0.0 Release"
   git push origin v1.0.0
   ```
3. **Create Release on GitHub:**
   - Navigate to `https://github.com/darkPegsus/Semiconductor-wafer-defect-detection/releases/new`.
   - Select tag: `v1.0.0`.
   - Release title: `WaferVision AI v1.0.0 — Production Release`.
   - Description: Include benchmark notes (Test Macro F1: 0.8746, Accuracy: 97.62%).
   - **Attach Binary Asset:** Drag and drop `models/checkpoints/exp1_custom_cnn_baseline_best.pt` into the assets box.
   - Click **Publish release**.
4. **Verify Anonymous Download:**
   ```bash
   curl -I -L https://github.com/darkPegsus/Semiconductor-wafer-defect-detection/releases/download/v1.0.0/exp1_custom_cnn_baseline_best.pt
   ```
   Confirm HTTP response `200 OK` (or `302 Found` redirecting to GitHub AWS S3 CDN).

---

## 5. Streamlit Community Cloud Setup Steps

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io) with your GitHub account.
2. Click **New app** (or **Create app**).
3. Fill in the repository settings:
   - **Repository:** `darkPegsus/Semiconductor-wafer-defect-detection`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **App URL (optional):** e.g., `wafervision-ai.streamlit.app`
4. Expand **Advanced settings**:
   - **Python version:** Select `3.12`.
   - **Secrets (optional):** None required. If using an alternative checkpoint mirror, specify:
     ```toml
     WAFER_CHECKPOINT_URL = "https://your-mirror.example.com/exp1_custom_cnn_baseline_best.pt"
     ```
5. Click **Deploy!**

---

## 6. Expected Startup Behavior

1. **Container Initialization:** Streamlit Cloud clones the repository and installs `requirements.txt`.
2. **Cold-Start Startup (First Visit):**
   - Streamlit launches `app/main.py`.
   - `@st.cache_resource` triggers `get_model()`.
   - `run_startup_healthcheck` detects missing local weights.
   - Status spinner displays: *"Verifying checkpoint and initializing CustomWaferCNN engine..."*
   - Checkpoint is downloaded (~2s for 6.28 MB) and verified.
   - Startup forward pass and Grad-CAM hooks execute smoke tests.
   - Health check logs: `Startup health check PASSED on cpu. Status: HEALTHY.`
3. **Application Ready:** The inspection dashboard displays with `Engine: CustomWaferCNN`, `Compute Device: CPU`, and green `Integrity: Verified` indicator.

---

## 7. Known Operational Limitations & Boundaries

- **CPU Compute Latency:** On Community Cloud shared vCPUs, forward inference takes ~1.8 ms per wafer, and full Grad-CAM activation generation takes ~18–25 ms.
- **Memory Ceiling:** Memory usage stays under 350 MB. Do not attempt multi-wafer batch training or full-dataset processing on Community Cloud.
- **Model Sensitivity:** In accordance with the Phase 7 benchmark:
  - Overall Test Macro F1: **0.8746**
  - Test Accuracy: **97.62%**
  - Scratch class recall is **64.12%** (39/170 escapes to 'none'). Suspected scratch tracks should always be visually inspected.
  - Loc cluster recall is **77.97%** (11.89% clean escape rate).
  - Grad-CAM heatmaps represent $8\times 8$ receptive downsampling of `conv_block4`, not exact pixel-level die segmentation.

---

## 8. Troubleshooting Guide

### Issue: "Checkpoint not found and automated retrieval failed (HTTP 404)"
- **Cause:** The release asset has not yet been uploaded to GitHub under tag `v1.0.0` or repository is private.
- **Solution:** Verify Step 4 above. Ensure the release is published (not draft) and the file is named exactly `exp1_custom_cnn_baseline_best.pt`.

### Issue: "Cryptographic integrity mismatch for checkpoint"
- **Cause:** Downloaded file was truncated, corrupted, or an incorrect `.pt` file was uploaded.
- **Solution:** Delete the corrupted file. Verify the local checkpoint SHA-256 against `b9df80c522dd4ac834945587becc53a31f92108c97e17afe222914017ddaa1f8` and re-upload to GitHub Releases.

### Issue: "Memory limit exceeded (OOM)"
- **Cause:** Container exceeded 1.0 GB RAM.
- **Solution:** Verify that `--extra-index-url https://download.pytorch.org/whl/cpu` is present in `requirements.txt`. Ensure no development dependencies (e.g. Jupyter, mypy) are included in production `requirements.txt`.
