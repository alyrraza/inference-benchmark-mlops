"""
InferBench FastAPI service.

Run locally with:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload

Endpoints:
    GET  /health   - liveness check, also reports which backends loaded
    POST /predict  - upload an image, get back a predicted class

This is Phase 2 of the build: FastAPI + the custom dynamic batching layer
only. There is no Redis cache, no PostgreSQL logging, and no Prometheus
/metrics endpoint yet - those are Phase 3, 4, and 5. Wiring them in later
means adding calls at the points marked in docs/sequence_diagram.puml,
not changing this file's structure.
"""

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app import config
from app.batching.models import QueueItem
from app.batching.worker import BatchWorker
from app.inference.base import InferenceBackend
from app.inference.registry import load_all_backends
from app.labels import label_for
from app.preprocessing import preprocess_image_bytes

# Shared state, populated at startup by the lifespan handler below.
# A plain asyncio.Queue is the hand-off point between request handlers
# (producers) and the BatchWorker (consumer) - see
# docs/concepts/02_async_queue_processing.md for why this specific
# primitive, not a list or a thread-safe queue.Queue.
request_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
backends: dict[str, InferenceBackend] = {}
worker: BatchWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backends, worker

    # Pinning PyTorch's thread count here (once, at process startup) rather
    # than leaving it at the library default keeps CPU usage predictable
    # and matches the thread budget the ONNX backend is pinned to
    # (app/inference/onnx_backend.py) - see config.py's TORCH_NUM_THREADS.
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    print("Loading inference backends (this happens once, not per-request)...")
    backends = load_all_backends()

    worker = BatchWorker(request_queue, backends)
    worker.start()
    print(f"BatchWorker started - window={config.BATCH_WINDOW_MS}ms, "
          f"max_batch_size={config.MAX_BATCH_SIZE}")

    yield

    print("Shutting down BatchWorker...")
    await worker.stop()


app = FastAPI(title="InferBench", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends_loaded": list(backends.keys()),
        "batch_window_ms": config.BATCH_WINDOW_MS,
        "max_batch_size": config.MAX_BATCH_SIZE,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    backend: str = Query(
        default=config.DEFAULT_BACKEND,
        description=f"Which inference backend to use. One of {config.AVAILABLE_BACKENDS}.",
    ),
):
    if backend not in config.AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{backend}'. Choose from {config.AVAILABLE_BACKENDS}.",
        )

    image_bytes = await file.read()
    try:
        image_tensor = preprocess_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    enqueued_at = time.perf_counter()
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    item = QueueItem(image=image_tensor, backend=backend, future=future, enqueued_at=enqueued_at)

    # This is the non-blocking hand-off from docs/sequence_diagram.puml:
    # "API -> API : await future (non-blocking)". Putting the item on the
    # queue returns immediately; awaiting the future is what actually
    # suspends this request handler (without blocking the event loop) until
    # the BatchWorker resolves it, in some other coroutine, later.
    await request_queue.put(item)

    try:
        logits = await future
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    total_latency_ms = (time.perf_counter() - enqueued_at) * 1000
    predicted_class_id = int(np.argmax(logits))

    return {
        "predicted_class_id": predicted_class_id,
        "predicted_label": label_for(predicted_class_id),
        "backend": backend,
        "batch_size": item.batch_size,
        "total_latency_ms": round(total_latency_ms, 2),
    }
