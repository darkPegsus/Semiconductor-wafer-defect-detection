# ruff: noqa: E402
"""Comprehensive hardware benchmark script for WaferVision AI on CPU and CUDA.

Measures granular pipeline latency across:
  1. Model loading & parameter freezing
  2. Raw input decoding (PNG and .npy)
  3. Preprocessing (Method C letterbox padding & normalization to [1, 1, 64, 64])
  4. Model-only inference latency (forward pass + softmax)
  5. Grad-CAM attribution latency (features.conv_block4 hooks + overlay generation)
  6. End-to-end upload-to-results latency (bytes -> decoding -> prep -> inference -> Grad-CAM)

Reports Mean, P50, P95, P99, Min, and Max metrics, distinguishing model-only inference
from full application response time.
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    DEFAULT_CHECKPOINT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    TARGET_GRAD_CAM_LAYER,
)
from app.services.explainability_service import WaferExplainabilityService
from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.model_loader import load_frozen_model


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute mean, P50, P95, P99, min, and max for latency series."""
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def benchmark_device(device_name: str, num_runs: int = 50, warmup_runs: int = 5) -> dict[str, Any]:
    """Execute complete granular benchmark suite on the specified torch device."""
    device = torch.device(device_name)
    print("\n" + "=" * 65)
    print(
        f" BENCHMARKING DEVICE: {device_name.upper()} ({num_runs} iterations, {warmup_runs} warm-ups)"
    )
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 1. Model Loading Time (Uncached / Cold Load from Disk)
    # -------------------------------------------------------------------------
    load_times = []
    import app.utils.model_loader as ml

    for _ in range(3):
        # Invalidate cache to test true disk load and SHA-256 verification
        ml._CACHED_MODEL = None
        ml._CACHED_METADATA = None
        ml._CACHED_DEVICE = None

        t0 = time.perf_counter()
        model, metadata, actual_device = load_frozen_model(
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            device=device,
            verify_hash=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        load_times.append((time.perf_counter() - t0) * 1000.0)

    model_load_time_ms = round(float(np.median(load_times)), 2)
    print(f"1. Model Load & Verify Time (Cold): {model_load_time_ms:.2f} ms")

    # -------------------------------------------------------------------------
    # Prepare Representative Raw Inputs
    # -------------------------------------------------------------------------
    # Create canonical wafer map with realistic defect pattern (Scratch)
    source_map = np.ones((54, 50), dtype=np.uint8)  # Normal wafer
    # Wafer circular boundary
    cy, cx = 27, 25
    y, x = np.ogrid[:54, :50]
    exterior = ((x - cx) ** 2 / (23**2) + (y - cy) ** 2 / (25**2)) > 1.0
    source_map[exterior] = 0
    # Add scratch defect streak
    for i in range(15, 38):
        source_map[i, min(49, 10 + (i - 15))] = 2

    # Render PNG bytes
    from app.utils.image_utils import wafer_to_rgb

    rgb_arr = wafer_to_rgb(source_map)
    pil_img = Image.fromarray(rgb_arr)
    png_buf = io.BytesIO()
    pil_img.save(png_buf, format="PNG")
    raw_png_bytes = png_buf.getvalue()

    # Render NPY bytes
    npy_buf = io.BytesIO()
    np.save(npy_buf, source_map)
    raw_npy_bytes = npy_buf.getvalue()

    # Preprocessed tensor for inference-only tests
    prep_2d, input_tensor = WaferPreprocessingService.preprocess_for_inference(source_map)
    tensor_dev = input_tensor.to(device)

    # -------------------------------------------------------------------------
    # Warm-up runs to prime CUDA kernels, caches, and memory allocations
    # -------------------------------------------------------------------------
    print(f"Executing {warmup_runs} warm-up cycles...")
    for _ in range(warmup_runs):
        with torch.inference_mode():
            _ = model(tensor_dev)
        _ = WaferPreprocessingService.load_wafer_from_bytes(raw_png_bytes, "test_wafer.png")
        _ = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=input_tensor,
            wafer_map_2d=prep_2d,
            target_class=7,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # 2. Input Decoding Latency (Raw Bytes -> Discrete Wafer Map)
    # -------------------------------------------------------------------------
    png_decoding_times = []
    npy_decoding_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = WaferPreprocessingService.load_wafer_from_bytes(raw_png_bytes, "test_wafer.png")
        png_decoding_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = WaferPreprocessingService.load_wafer_from_bytes(raw_npy_bytes, "test_wafer.npy")
        npy_decoding_times.append((time.perf_counter() - t0) * 1000.0)

    png_decode_stats = compute_percentiles(png_decoding_times)
    npy_decode_stats = compute_percentiles(npy_decoding_times)
    print(
        f"2. Input Decoding (PNG):  Mean {png_decode_stats['mean_ms']:.3f} ms | P50 {png_decode_stats['p50_ms']:.3f} ms | P95 {png_decode_stats['p95_ms']:.3f} ms"
    )
    print(
        f"   Input Decoding (NPY):  Mean {npy_decode_stats['mean_ms']:.3f} ms | P50 {npy_decode_stats['p50_ms']:.3f} ms | P95 {npy_decode_stats['p95_ms']:.3f} ms"
    )

    # -------------------------------------------------------------------------
    # 3. Preprocessing Latency (Method C Letterbox to [1, 1, 64, 64])
    # -------------------------------------------------------------------------
    preprocessing_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = WaferPreprocessingService.preprocess_for_inference(source_map)
        preprocessing_times.append((time.perf_counter() - t0) * 1000.0)

    prep_stats = compute_percentiles(preprocessing_times)
    print(
        f"3. Method C Preprocessing: Mean {prep_stats['mean_ms']:.3f} ms | P50 {prep_stats['p50_ms']:.3f} ms | P95 {prep_stats['p95_ms']:.3f} ms"
    )

    # -------------------------------------------------------------------------
    # 4. Model-Only Inference Latency (Forward Pass + Softmax)
    # -------------------------------------------------------------------------
    model_inference_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor_dev)
            _ = torch.softmax(logits, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_inference_times.append((time.perf_counter() - t0) * 1000.0)

    model_inf_stats = compute_percentiles(model_inference_times)
    print(
        f"4. Model-Only Inference:   Mean {model_inf_stats['mean_ms']:.3f} ms | P50 {model_inf_stats['p50_ms']:.3f} ms | P95 {model_inf_stats['p95_ms']:.3f} ms | P99 {model_inf_stats['p99_ms']:.3f} ms"
    )

    # -------------------------------------------------------------------------
    # 5. Grad-CAM Attribution Latency
    # -------------------------------------------------------------------------
    gradcam_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=input_tensor,
            wafer_map_2d=prep_2d,
            target_class=7,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        gradcam_times.append((time.perf_counter() - t0) * 1000.0)

    gradcam_stats = compute_percentiles(gradcam_times)
    print(
        f"5. Grad-CAM Latency:       Mean {gradcam_stats['mean_ms']:.3f} ms | P50 {gradcam_stats['p50_ms']:.3f} ms | P95 {gradcam_stats['p95_ms']:.3f} ms"
    )

    # -------------------------------------------------------------------------
    # 6. Complete End-to-End Upload-to-Results Pipeline Latency
    # -------------------------------------------------------------------------
    e2e_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step A: Decode uploaded raw image bytes
        decoded_map = WaferPreprocessingService.load_wafer_from_bytes(
            raw_png_bytes, "test_wafer.png"
        )
        # Step B: Method C Preprocessing
        prep_map, tensor_in = WaferPreprocessingService.preprocess_for_inference(decoded_map)
        # Step C: Model Inference
        pred_result = WaferInferenceService.predict(
            model=model, input_tensor=tensor_in, device=device
        )
        # Step D: Grad-CAM Explainability
        _ = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=tensor_in,
            wafer_map_2d=prep_map,
            target_class=pred_result["predicted_class_idx"],
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    e2e_stats = compute_percentiles(e2e_times)
    print(
        f"6. End-to-End App Latency: Mean {e2e_stats['mean_ms']:.3f} ms | P50 {e2e_stats['p50_ms']:.3f} ms | P95 {e2e_stats['p95_ms']:.3f} ms | P99 {e2e_stats['p99_ms']:.3f} ms"
    )

    # -------------------------------------------------------------------------
    # Memory Usage
    # -------------------------------------------------------------------------
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    if device.type == "cuda":
        gpu_allocated_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        gpu_reserved_mb = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2)
        print(
            f"Peak GPU Memory:          {gpu_allocated_mb:.2f} MB allocated, {gpu_reserved_mb:.2f} MB reserved"
        )

    return {
        "device": device_name,
        "iterations": num_runs,
        "warmup_runs": warmup_runs,
        "model_load_time_ms": model_load_time_ms,
        "input_decoding_png": png_decode_stats,
        "input_decoding_npy": npy_decode_stats,
        "method_c_preprocessing": prep_stats,
        "model_only_inference": model_inf_stats,
        "grad_cam_attribution": gradcam_stats,
        "end_to_end_pipeline": e2e_stats,
        "gpu_allocated_mb": gpu_allocated_mb,
        "gpu_reserved_mb": gpu_reserved_mb,
    }


def main():
    print("=" * 65)
    print(" WAFERVISION AI - GRANULAR HARDWARE BENCHMARK SUITE")
    print("=" * 65)

    benchmark_data = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "model_architecture": "CustomWaferCNN",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "target_layer": TARGET_GRAD_CAM_LAYER,
        },
        "devices": {},
    }

    # Benchmark CPU
    benchmark_data["devices"]["cpu"] = benchmark_device("cpu", num_runs=50, warmup_runs=5)

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        benchmark_data["devices"]["cuda"] = benchmark_device("cuda", num_runs=50, warmup_runs=5)
    else:
        print("\nCUDA accelerator not detected.")

    out_path = PROJECT_ROOT / "docs" / "deployment" / "hardware_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Granular benchmark results saved to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
