"""
Demonstrates a cache MISS followed by a cache HIT against the running
service, with real latency numbers - the actual proof that Redis caching
works, not just that the code compiles.

Requires the service to already be running (see app/main.py's docstring
for the uvicorn command) and Redis to be reachable (see app/config.py's
REDIS_HOST/REDIS_PORT).

Usage:
    .venv/Scripts/python.exe scripts/verify_cache.py
"""

import io
import time

import httpx
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"
BACKEND = "pytorch"


def make_fixed_test_image_bytes() -> bytes:
    # Fixed seed - this must produce the exact same bytes every run. The
    # cache key is sha256(these bytes), so the two requests below only
    # land on the same cache entry because they send the literal same
    # bytes object, not just "a similar-looking image".
    rng = np.random.default_rng(seed=7)
    array = (rng.random((224, 224, 3)) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")  # lossless, though byte-identity here comes from reusing the same buffer, not the format
    return buffer.getvalue()


def send_request(client: httpx.Client, image_bytes: bytes, label: str) -> dict:
    files = {"file": ("test.png", image_bytes, "image/png")}
    start = time.perf_counter()
    response = client.post(API_URL, files=files, params={"backend": BACKEND})
    client_latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    data = response.json()
    print(f"{label}: cache_hit={data['cache_hit']!s:<5}  "
          f"server_latency_ms={data['total_latency_ms']:>8.2f}  "
          f"client_latency_ms={client_latency_ms:>8.2f}  "
          f"predicted_label={data['predicted_label']}")
    return data


def main():
    image_bytes = make_fixed_test_image_bytes()

    with httpx.Client(timeout=30) as client:
        print("Request 1 - expect a cache MISS (goes through the batching worker and the model):")
        miss_result = send_request(client, image_bytes, "MISS")

        print("\nRequest 2 - identical image bytes - expect a cache HIT (served from Redis):")
        hit_result = send_request(client, image_bytes, "HIT ")

    assert miss_result["cache_hit"] is False, "expected the first request to be a cache miss"
    assert hit_result["cache_hit"] is True, "expected the second request to be a cache hit"
    assert miss_result["predicted_class_id"] == hit_result["predicted_class_id"], \
        "cached prediction should exactly match the original"

    speedup = miss_result["total_latency_ms"] / max(hit_result["total_latency_ms"], 0.01)
    print(f"\nCache hit was {speedup:.1f}x faster than the original cache miss "
          f"({miss_result['total_latency_ms']:.2f}ms -> {hit_result['total_latency_ms']:.2f}ms).")


if __name__ == "__main__":
    main()
