"""
Fires a burst of concurrent requests at the running FastAPI service to
prove the dynamic batching layer actually groups them - this is the
hands-on verification for the "build from scratch" batching centerpiece.

Requires the service to already be running (see app/main.py's docstring
for the uvicorn command). Uses synthetic random images so no dataset
download is needed.

Usage:
    .venv/Scripts/python.exe scripts/load_test.py
"""

import asyncio
import io
import time

import httpx
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 20
BACKEND = "pytorch"


def make_fake_image_bytes() -> bytes:
    # Random pixels are enough here - the batching layer only cares about
    # tensor shape and arrival timing, not image content.
    array = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    image = Image.fromarray(array)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


async def send_one(client: httpx.AsyncClient, index: int) -> dict:
    image_bytes = make_fake_image_bytes()
    files = {"file": (f"test_{index}.jpg", image_bytes, "image/jpeg")}
    start = time.perf_counter()
    response = await client.post(API_URL, files=files, params={"backend": BACKEND})
    client_latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return {"index": index, "client_latency_ms": client_latency_ms, **response.json()}


async def main():
    print(f"Firing {NUM_REQUESTS} concurrent requests at {API_URL} (backend={BACKEND})...")

    async with httpx.AsyncClient(timeout=60) as client:
        start = time.perf_counter()
        # asyncio.gather launches all requests essentially at once - this is
        # what gives the batch worker a chance to actually group them,
        # instead of handling one request at a time in sequence.
        results = await asyncio.gather(*(send_one(client, i) for i in range(NUM_REQUESTS)))
        total_wall_ms = (time.perf_counter() - start) * 1000

    batch_sizes = sorted(r["batch_size"] for r in results)
    server_latencies = [r["total_latency_ms"] for r in results]

    print(f"\nAll {NUM_REQUESTS} requests completed in {total_wall_ms:.1f}ms wall-clock time")
    print(f"Batch sizes the worker actually formed per request: {batch_sizes}")
    print(f"Server-reported latency - mean: {np.mean(server_latencies):.1f}ms, "
          f"min: {min(server_latencies):.1f}ms, max: {max(server_latencies):.1f}ms")

    if any(bs > 1 for bs in batch_sizes):
        print("\nBatching worked: at least one request shared a forward pass with others.")
    else:
        print("\nNo requests were batched together (all batch_size=1) - "
              "either traffic wasn't concurrent enough, or the window is too short.")


if __name__ == "__main__":
    asyncio.run(main())
