# ruff: noqa: E402
"""Hardware benchmark script for CustomWaferCNN inference and Grad-CAM on CPU and CUDA."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.explainability_service import WaferExplainabilityService
from app.services.inference_service import WaferInferenceService
from app.services.preprocessing_service import WaferPreprocessingService
from app.utils.model_loader import load_frozen_model


def benchmark_device(device_name: str, num_runs: int = 50) -> dict:
    device = torch.device(device_name)
    print("\n========================================================")
    print(f" BENCHMARKING DEVICE: {device_name.upper()}")
    print("========================================================")

    # 1. Model Loading Time
    t0 = time.perf_counter()
    model, metadata, actual_device = load_frozen_model(device=device)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Model Load Time: {load_time_ms:.2f} ms")

    # 2. Prepare Canonical Test Input [1, 1, 64, 64]
    sample_path = PROJECT_ROOT / "tests" / "data" / "roundtrip_sources" / "near_full_original.npy"
    if sample_path.exists():
        raw_map = np.load(sample_path)
    else:
        raw_map = np.ones((50, 43), dtype=np.uint8)

    prep_2d, tensor_input = WaferPreprocessingService.preprocess_for_inference(raw_map)
    tensor_dev = tensor_input.to(device)

    # Warmup runs
    for _ in range(5):
        with torch.inference_mode():
            _ = model(tensor_dev)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 3. Inference Latency Benchmark
    inference_latencies = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        res = WaferInferenceService.predict(model=model, input_tensor=tensor_input, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        inference_latencies.append((t_end - t_start) * 1000.0)

    inf_mean = float(np.mean(inference_latencies))
    inf_p50 = float(np.median(inference_latencies))
    inf_p95 = float(np.percentile(inference_latencies, 95))
    inf_p99 = float(np.percentile(inference_latencies, 99))
    inf_min = float(np.min(inference_latencies))
    inf_max = float(np.max(inference_latencies))

    print(f"Inference Latency ({num_runs} runs):")
    print(
        f"  Mean: {inf_mean:.3f} ms | P50: {inf_p50:.3f} ms | P95: {inf_p95:.3f} ms | P99: {inf_p99:.3f} ms (Min: {inf_min:.3f} ms, Max: {inf_max:.3f} ms)"
    )

    # 4. Grad-CAM Latency Benchmark
    gradcam_latencies = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        _ = WaferExplainabilityService.generate_explanation(
            model=model,
            input_tensor=tensor_input,
            wafer_map_2d=prep_2d,
            target_class=res["predicted_class_idx"],
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        gradcam_latencies.append((t_end - t_start) * 1000.0)

    gcam_mean = float(np.mean(gradcam_latencies))
    gcam_p50 = float(np.median(gradcam_latencies))
    gcam_p95 = float(np.percentile(gradcam_latencies, 95))

    print(f"Grad-CAM Latency ({num_runs} runs):")
    print(f"  Mean: {gcam_mean:.3f} ms | P50: {gcam_p50:.3f} ms | P95: {gcam_p95:.3f} ms")

    # 5. Memory Usage
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    if device.type == "cuda":
        gpu_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        print(
            f"Peak GPU Memory: {gpu_allocated_mb:.2f} MB allocated, {gpu_reserved_mb:.2f} MB reserved"
        )

    return {
        "device": device_name,
        "load_time_ms": round(load_time_ms, 2),
        "inference_mean_ms": round(inf_mean, 3),
        "inference_p50_ms": round(inf_p50, 3),
        "inference_p95_ms": round(inf_p95, 3),
        "inference_p99_ms": round(inf_p99, 3),
        "gradcam_mean_ms": round(gcam_mean, 3),
        "gradcam_p50_ms": round(gcam_p50, 3),
        "gradcam_p95_ms": round(gcam_p95, 3),
        "gpu_allocated_mb": round(gpu_allocated_mb, 2),
        "gpu_reserved_mb": round(gpu_reserved_mb, 2),
    }


def main():
    results = {}
    # Benchmark CPU
    results["cpu"] = benchmark_device("cpu", num_runs=50)

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        results["cuda"] = benchmark_device("cuda", num_runs=50)
    else:
        print("\nCUDA is not available on this machine.")

    import json

    out_path = PROJECT_ROOT / "docs" / "deployment" / "hardware_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved hardware benchmarks to: {out_path}")


if __name__ == "__main__":
    main()
