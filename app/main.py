"""
InferBench FastAPI service.

Run locally with:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload

Endpoints:
    GET  /health   - liveness check, also reports which backends loaded
    POST /predict  - upload an image, get back a predicted class

This is Phase 5 of the build: FastAPI + the custom dynamic batching layer
(Phase 2) + Redis response caching (Phase 3) + PostgreSQL request logging
(Phase 4) + Prometheus /metrics (Phase 5).
"""

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import cache, config, db, metrics
from app.batching.models import QueueItem
from app.batching.worker import BatchWorker
from app.inference.registry import load_all_backends
from app.labels import label_for
from app.preprocessing import preprocess_image_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # request_queue is created HERE, inside lifespan, not as a module-level
    # global - an asyncio.Queue binds itself to whichever event loop is
    # currently running at the moment it's constructed. A module-level
    # `asyncio.Queue()` gets bound to the loop that happened to exist when
    # the module was first imported; if the app's lifespan ever starts
    # again under a *different* loop (this bit us running pytest: each
    # test's TestClient spins up its own event loop), every
    # `await request_queue.put(...)` call fails with "Queue ... is bound to
    # a different event loop". Creating it inside lifespan() guarantees
    # it's always bound to the loop actually running this app instance.
    # app.state is FastAPI's built-in place for exactly this kind of
    # request-accessible, lifespan-scoped state.
    app.state.request_queue = asyncio.Queue()

    # Pinning PyTorch's thread count here (once, at process startup) rather
    # than leaving it at the library default keeps CPU usage predictable
    # and matches the thread budget the ONNX backend is pinned to
    # (app/inference/onnx_backend.py) - see config.py's TORCH_NUM_THREADS.
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    print("Loading inference backends (this happens once, not per-request)...")
    app.state.backends = load_all_backends()

    app.state.worker = BatchWorker(app.state.request_queue, app.state.backends)
    app.state.worker.start()
    print(f"BatchWorker started - window={config.BATCH_WINDOW_MS}ms, "
          f"max_batch_size={config.MAX_BATCH_SIZE}")

    print("Connecting to PostgreSQL request_log store...")
    try:
        app.state.db_pool = await db.create_pool()
        print("Connected - request_log table ready.")
    except Exception as exc:
        # Same graceful-degradation stance as Redis: the metadata store is
        # not required for correct predictions, so a database that's down
        # at startup shouldn't prevent the service itself from starting.
        print(f"[db] could not connect at startup, request logging disabled: {exc}")
        app.state.db_pool = None

    yield

    print("Shutting down BatchWorker...")
    await app.state.worker.stop()
    if app.state.db_pool is not None:
        await app.state.db_pool.close()


app = FastAPI(title="InferBench", lifespan=lifespan)

# Local-dev-only: lets the Vite frontend (frontend/, a different origin
# from this API) call /predict directly from the browser. See
# config.py's CORS_ALLOWED_ORIGINS and docs/concepts/05c_demo_frontend.md
# for why this is scoped narrowly to known local origins rather than "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "backends_loaded": list(request.app.state.backends.keys()),
        "batch_window_ms": config.BATCH_WINDOW_MS,
        "max_batch_size": config.MAX_BATCH_SIZE,
        "redis_available": cache.is_redis_available(),
        "db_available": request.app.state.db_pool is not None,
    }


@app.get("/metrics")
async def metrics_endpoint():
    # Prometheus scrapes this endpoint on its own schedule (see
    # prometheus/prometheus.yml's scrape_interval) - this service never
    # pushes anything to Prometheus, it only exposes the current state of
    # its in-process counters/histograms whenever asked. Content-Type
    # matters here: Prometheus's scraper expects this exact format.
    return Response(
        content=metrics.latest_metrics_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/cache/stats")
async def cache_stats():
    """
    Reports cache health and Redis's own built-in hit/miss counters.

    Deliberately uses Redis's INFO command instead of a KEYS
    inferbench:predict:* scan to count entries - KEYS walks the entire
    keyspace and blocks Redis while it does, which is a real production
    footgun on a busy instance. INFO's keyspace_hits/keyspace_misses
    counters are O(1) to read and don't touch the keyspace at all.
    """
    if not cache.is_redis_available():
        return {"redis_available": False}

    stats = cache.get_client().info("stats")
    return {
        "redis_available": True,
        "keyspace_hits": stats.get("keyspace_hits"),
        "keyspace_misses": stats.get("keyspace_misses"),
        "cache_ttl_seconds": config.CACHE_TTL_SECONDS,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    backend: str = Query(
        default=config.DEFAULT_BACKEND,
        description=f"Which inference backend to use. One of {config.AVAILABLE_BACKENDS}.",
    ),
):
    backends = request.app.state.backends
    request_queue = request.app.state.request_queue

    if backend not in config.AVAILABLE_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{backend}'. Choose from {config.AVAILABLE_BACKENDS}.",
        )
    if backend not in backends:
        # Recognized backend name, but load_all_backends() couldn't load it
        # on this particular run (e.g. missing model artifact) - a 503,
        # not a 400, since the request itself was valid, the server just
        # can't currently serve it.
        raise HTTPException(
            status_code=503,
            detail=f"Backend '{backend}' is not currently loaded. Available right now: {list(backends.keys())}.",
        )

    image_bytes = await file.read()
    enqueued_at = time.perf_counter()

    # This is docs/sequence_diagram.puml's "API -> Redis : check cache for
    # identical request" step. It happens before preprocessing on purpose -
    # a cache hit skips the resize/normalize work too, not just the model
    # call, since neither is needed to answer a request we've already
    # answered before.
    cached = cache.get_cached_prediction(image_bytes, backend)
    if cached is not None:
        total_latency_ms = (time.perf_counter() - enqueued_at) * 1000

        # "API -> DB : log latency, batch_size, model_type" - drawn in the
        # sequence diagram AFTER the alt/else block closes, meaning both
        # the cache-hit and cache-miss paths log here, not just misses.
        # A cache hit's row has batch_size = NULL - there was no batch,
        # nothing to log there - but it's still a real request worth
        # counting for latency/throughput analysis later.
        if request.app.state.db_pool is not None:
            await db.log_request(
                request.app.state.db_pool, backend, True, None,
                cached["predicted_class_id"], total_latency_ms,
            )
        metrics.record_request(backend, True, total_latency_ms)

        return {
            **cached,
            "backend": backend,
            "cache_hit": True,
            "batch_size": None,
            "total_latency_ms": round(total_latency_ms, 2),
        }

    # Cache miss from here on - this is the "else Cache Miss" branch of the
    # sequence diagram: preprocess, enqueue, await the batch worker's
    # result, then store it in Redis before returning.
    try:
        image_tensor = preprocess_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

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

    predicted_class_id = int(np.argmax(logits))
    result = {
        "predicted_class_id": predicted_class_id,
        "predicted_label": label_for(predicted_class_id),
    }

    # "API -> Redis : store result in cache" - only real inference results
    # get cached, never a cache hit re-caching itself.
    cache.store_prediction(image_bytes, backend, result)

    total_latency_ms = (time.perf_counter() - enqueued_at) * 1000

    if request.app.state.db_pool is not None:
        await db.log_request(
            request.app.state.db_pool, backend, False, item.batch_size,
            predicted_class_id, total_latency_ms,
        )
    metrics.record_request(backend, False, total_latency_ms)

    return {
        **result,
        "backend": backend,
        "cache_hit": False,
        "batch_size": item.batch_size,
        "total_latency_ms": round(total_latency_ms, 2),
    }
