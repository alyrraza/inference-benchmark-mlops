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

# --- Redis cache settings ---
# Port 6380, not Redis's default 6379: this dev machine has a pre-existing
# Windows service already bound to 6379 that couldn't be removed (needed
# admin rights this environment doesn't have), so the project's own Redis
# instance runs on 6380 instead, with its own data directory, to stay
# completely separate from that leftover service.
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))
REDIS_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("REDIS_CONNECT_TIMEOUT_SECONDS", "1.0"))

# How long a cached prediction stays valid. Predictions are deterministic
# (same image + same backend always produces the same result), so nothing
# ever goes "stale" in the usual web-cache sense - this TTL exists to
# bound memory growth and to self-heal within an hour if a bad entry were
# ever cached, not because the data expires in any meaningful way.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

# --- PostgreSQL metadata store settings ---
# Port 5433, not Postgres's default 5432: same reasoning as Redis's 6380 -
# this dev machine has leftover PostgreSQL install remnants (data
# directories with no server binaries attached to them) from a previous
# setup, and running on a distinct port keeps this project's own instance
# unambiguous and independent of whatever else might already be configured
# to expect port 5432 on this machine.
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "5433"))
DB_NAME = os.environ.get("DB_NAME", "inferbench")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")  # empty - local dev cluster uses trust auth, not for production

# Connection pool sizing - see app/db.py and docs/concepts/04_postgres_metadata_store.md
# for why every request reuses a pooled connection instead of opening a
# fresh one each time.
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "10"))

# --- CORS (local demo frontend only) ---
# The Vite dev server (frontend/) runs on a different origin
# (http://localhost:5173) than this API (http://127.0.0.1:8000), so the
# browser blocks frontend fetch() calls to /predict unless the API
# explicitly allows it. This is a local-development convenience only -
# see docs/concepts/05c_demo_frontend.md for why a real deployment would
# restrict this to specific known origins, not read from an env var that
# defaults to "allow everything localhost."
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
