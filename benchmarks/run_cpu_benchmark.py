"""
CPU benchmark: PyTorch (eager) vs ONNX Runtime (CPU) vs TorchScript.

Model: google/vit-base-patch16-224, same weights used in the Kaggle GPU
benchmark (kaggle/results/benchmark_results.json). This script does NOT
redo that GPU work - it produces a separate, comparable CPU-only result set
using the exact same methodology (warmup, 50 timed runs per batch size,
p50/p95/p99), so the two JSON files can be placed side by side later in the
Gradio demo.

METHODOLOGY NOTE (read this before trusting the numbers):
The Kaggle notebook's TensorRT benchmark ran one untimed "priming" call
before timing each new batch size, because TensorRT compiles a distinct
GPU engine per input shape. PyTorch-CPU and ONNX-Runtime-CPU don't compile
engines, but they do lazily pick CPU kernels / allocate buffers on the
first call for a shape they haven't seen yet. To keep the comparison fair
across all three CPU backends, this script primes every batch size once
(one untimed call) before the 50 timed runs, matching what the GPU script
did for TensorRT. See docs/concepts/00_cpu_vs_gpu_inference.md for why this
matters.
"""

import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"
# NOTE: kaggle/vit_base.onnx can't be reused here - it's only the graph
# shell, its weight data file (vit_base.onnx.data) never left Kaggle. This
# is a fresh export with the same settings (opset 17, dynamic batch axis),
# same weights, produced by benchmarks/export_onnx.py. See that file's
# docstring for details.
ONNX_PATH = "benchmarks/vit_base_cpu.onnx"
TORCHSCRIPT_PATH = "benchmarks/vit_base_torchscript.pt"
OUTPUT_PATH = "benchmarks/results/cpu_benchmark_results.json"
BATCH_SIZES = [1, 4, 8, 16]
NUM_RUNS = 50
NUM_WARMUP = 10


def load_models():
    print("Loading PyTorch model (eager mode)...")
    pytorch_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    pytorch_model.eval()

    print("Loading TorchScript model...")
    torchscript_model = torch.jit.load(TORCHSCRIPT_PATH)
    torchscript_model.eval()

    print("Loading ONNX Runtime session (CPUExecutionProvider)...")
    # Explicitly restricting to CPUExecutionProvider even though this machine
    # has no GPU anyway - this makes the benchmark's intent unambiguous to
    # anyone reading the code later.
    onnx_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print("ONNX active provider:", onnx_session.get_providers())

    return pytorch_model, torchscript_model, onnx_session


def percentile_stats(latencies_ms):
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def benchmark_pytorch(model, batch_size, num_runs=NUM_RUNS):
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Prime this shape once, untimed (see module docstring)
    with torch.no_grad():
        _ = model(input_tensor)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {"batch_size": batch_size, **percentile_stats(latencies)}


def benchmark_torchscript(model, batch_size, num_runs=NUM_RUNS):
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        _ = model(input_tensor)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {"batch_size": batch_size, **percentile_stats(latencies)}


def benchmark_onnx(session, batch_size, num_runs=NUM_RUNS):
    input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    _ = session.run(None, {"pixel_values": input_data})

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {"pixel_values": input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {"batch_size": batch_size, **percentile_stats(latencies)}


def run_warmup(pytorch_model, torchscript_model, onnx_session):
    print(f"Warming up all three backends ({NUM_WARMUP} runs each, batch_size=1)...")
    warmup_torch = torch.randn(1, 3, 224, 224)
    warmup_np = warmup_torch.numpy()

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = pytorch_model(warmup_torch)
            _ = torchscript_model(warmup_torch)

    for _ in range(NUM_WARMUP):
        _ = onnx_session.run(None, {"pixel_values": warmup_np})

    print("Warmup done.")


def validate_accuracy(pytorch_model, torchscript_model, onnx_session, num_samples=20):
    """
    Compares logits across all three backends on the same synthetic inputs.
    We use fixed-seed random "images" instead of a downloaded dataset because
    this check validates numerical consistency between export formats (do
    ONNX/TorchScript produce the same answer as the original PyTorch model
    given identical weights?), not classification accuracy against real
    labels - that was already covered by the Kaggle GPU notebook's
    tiny-imagenet accuracy check.
    """
    print(f"Validating accuracy on {num_samples} synthetic samples...")
    rng = np.random.default_rng(seed=42)

    matches_onnx = 0
    matches_torchscript = 0
    diffs_onnx = []
    diffs_torchscript = []

    with torch.no_grad():
        for _ in range(num_samples):
            image_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
            image_torch = torch.from_numpy(image_np)

            pytorch_logits = pytorch_model(image_torch).logits.numpy()
            torchscript_logits = torchscript_model(image_torch)
            # torch.jit.trace flattens HF's ModelOutput into a plain dict,
            # so attribute access (.logits) no longer works post-tracing -
            # only the eager PyTorch model still returns the ModelOutput object.
            if isinstance(torchscript_logits, dict):
                torchscript_logits = torchscript_logits["logits"]
            elif hasattr(torchscript_logits, "logits"):
                torchscript_logits = torchscript_logits.logits
            torchscript_logits = torchscript_logits.numpy()
            onnx_logits = onnx_session.run(None, {"pixel_values": image_np})[0]

            if np.argmax(pytorch_logits) == np.argmax(onnx_logits):
                matches_onnx += 1
            if np.argmax(pytorch_logits) == np.argmax(torchscript_logits):
                matches_torchscript += 1

            diffs_onnx.append(np.mean(np.abs(pytorch_logits - onnx_logits)))
            diffs_torchscript.append(np.mean(np.abs(pytorch_logits - torchscript_logits)))

    return {
        "test_set": f"synthetic fixed-seed random tensors, {num_samples} samples (seed=42)",
        "onnx_vs_pytorch_match_pct": round(100 * matches_onnx / num_samples, 2),
        "torchscript_vs_pytorch_match_pct": round(100 * matches_torchscript / num_samples, 2),
        "onnx_vs_pytorch_mean_abs_diff": float(np.mean(diffs_onnx)),
        "torchscript_vs_pytorch_mean_abs_diff": float(np.mean(diffs_torchscript)),
    }


def build_speedup_summary(pytorch_results, onnx_results, torchscript_results):
    summary = {}
    for pt, ox, ts in zip(pytorch_results, onnx_results, torchscript_results):
        bs = pt["batch_size"]
        summary[f"batch_{bs}"] = {
            "pytorch_ms": pt["mean_ms"],
            "onnx_ms": ox["mean_ms"],
            "torchscript_ms": ts["mean_ms"],
            "onnx_speedup_vs_pytorch": round(pt["mean_ms"] / ox["mean_ms"], 2),
            "torchscript_speedup_vs_pytorch": round(pt["mean_ms"] / ts["mean_ms"], 2),
        }
    return summary


def main():
    pytorch_model, torchscript_model, onnx_session = load_models()
    total_params = sum(p.numel() for p in pytorch_model.parameters())

    run_warmup(pytorch_model, torchscript_model, onnx_session)

    print("\nBenchmarking PyTorch (CPU, eager)...")
    pytorch_results = []
    for bs in BATCH_SIZES:
        result = benchmark_pytorch(pytorch_model, bs)
        pytorch_results.append(result)
        print(f"  Batch {bs:2d} | Mean: {result['mean_ms']:.2f}ms | P50: {result['p50_ms']:.2f}ms | "
              f"P95: {result['p95_ms']:.2f}ms | P99: {result['p99_ms']:.2f}ms")

    print("\nBenchmarking ONNX Runtime (CPU)...")
    onnx_results = []
    for bs in BATCH_SIZES:
        result = benchmark_onnx(onnx_session, bs)
        onnx_results.append(result)
        print(f"  Batch {bs:2d} | Mean: {result['mean_ms']:.2f}ms | P50: {result['p50_ms']:.2f}ms | "
              f"P95: {result['p95_ms']:.2f}ms | P99: {result['p99_ms']:.2f}ms")

    print("\nBenchmarking TorchScript (CPU)...")
    torchscript_results = []
    for bs in BATCH_SIZES:
        result = benchmark_torchscript(torchscript_model, bs)
        torchscript_results.append(result)
        print(f"  Batch {bs:2d} | Mean: {result['mean_ms']:.2f}ms | P50: {result['p50_ms']:.2f}ms | "
              f"P95: {result['p95_ms']:.2f}ms | P99: {result['p99_ms']:.2f}ms")

    accuracy_validation = validate_accuracy(pytorch_model, torchscript_model, onnx_session)
    speedup_summary = build_speedup_summary(pytorch_results, onnx_results, torchscript_results)

    output = {
        "metadata": {
            "model": MODEL_NAME,
            "total_parameters": total_params,
            "device": "CPU",
            "cpu": platform.processor() or platform.machine(),
            "os": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "script": "benchmarks/run_cpu_benchmark.py",
        },
        "pytorch_baseline": pytorch_results,
        "onnx_runtime_cpu": onnx_results,
        "torchscript": torchscript_results,
        "accuracy_validation": accuracy_validation,
        "speedup_summary": speedup_summary,
        "environment_issues_resolved": [],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
