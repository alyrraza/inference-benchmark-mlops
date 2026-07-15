"""
Central configuration for the InferBench FastAPI service.

Every other module in app/ imports settings from here instead of hardcoding
paths or magic numbers. This is the single place you'd touch to, say, widen
the batching window or point at a different model.
"""

import os
from pathlib import Path

# Resolve paths relative to the project root (two levels up from this file:
# app/config.py -> app/ -> project root), so the service works no matter
# what directory you launch uvicorn from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse the same local Hugging Face cache from Phase 1 instead of the
# default C: drive cache. setdefault() means an explicit `export HF_HOME=...`
# before launching still wins - this is just a safety net so the service
# does the right thing even if you forget to set it.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))

MODEL_NAME = "google/vit-base-patch16-224"

# Reuse the exact model artifacts exported in Phase 1 rather than
# re-exporting them - same weights, already validated for numerical
# consistency against eager PyTorch (see benchmarks/results/cpu_benchmark_results.json).
ONNX_MODEL_PATH = str(PROJECT_ROOT / "benchmarks" / "vit_base_cpu.onnx")
TORCHSCRIPT_MODEL_PATH = str(PROJECT_ROOT / "benchmarks" / "vit_base_torchscript.pt")

# --- Dynamic batching settings ---
# How long the batch worker waits, after the FIRST request in a new batch
# arrives, before it stops collecting more requests and runs inference.
# This is the "10ms window" from docs/sequence_diagram.puml.
BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "10"))
BATCH_WINDOW_SECONDS = BATCH_WINDOW_MS / 1000

# Hard cap on how many requests go into one forward pass, even if more
# arrive within the window. Without this, a traffic spike could build an
# arbitrarily large batch and blow up memory/latency in one shot.
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))

# --- Inference backend settings ---
AVAILABLE_BACKENDS = ["pytorch", "onnx", "torchscript"]
DEFAULT_BACKEND = os.environ.get("DEFAULT_BACKEND", "pytorch")

# Pinned explicitly (rather than left at each library's own default) so
# PyTorch and ONNX Runtime use the same CPU thread budget. Phase 1's
# benchmark flagged unpinned thread counts as a fairness gap between
# backends - see docs/concepts/00_cpu_vs_gpu_inference.md, "common mistakes".
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
