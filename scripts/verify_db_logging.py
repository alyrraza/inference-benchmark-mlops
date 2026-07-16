"""
Demonstrates that /predict requests actually get logged to PostgreSQL -
real proof, not just reading the code and assuming it works.

Sends a few requests against a running service (mixing backends and
including a repeat to trigger a cache hit), then queries the request_log
table directly and prints the rows - the same table a future Grafana
dashboard (Phase 5) would read from.

Requires the service to already be running (see app/main.py's docstring
for the uvicorn command) and PostgreSQL to be reachable (see
app/config.py's DB_HOST/DB_PORT).

Usage:
    .venv/Scripts/python.exe scripts/verify_db_logging.py
"""

import asyncio
import io
import os

import asyncpg
import httpx
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

# Matches app/config.py's defaults - duplicated here rather than imported
# so this script stays a standalone file runnable as `python
# scripts/verify_db_logging.py` without needing the project root on
# sys.path, same as scripts/load_test.py and scripts/verify_cache.py.
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "5433"))
DB_NAME = os.environ.get("DB_NAME", "inferbench")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")


def make_image_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed=seed)
    array = (rng.random((224, 224, 3)) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def send_request(client: httpx.Client, image_bytes: bytes, backend: str) -> dict:
    files = {"file": ("test.png", image_bytes, "image/png")}
    response = client.post(API_URL, files=files, params={"backend": backend})
    response.raise_for_status()
    return response.json()


async def fetch_recent_rows(limit: int = 10) -> list[asyncpg.Record]:
    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        min_size=1,
        max_size=1,
    )
    try:
        async with pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT id, created_at, backend, cache_hit, batch_size,
                       predicted_class_id, total_latency_ms
                FROM request_log
                ORDER BY id DESC
                LIMIT $1
                """,
                limit,
            )
    finally:
        await pool.close()


def main():
    print(f"Sending requests to {API_URL}...")

    image_a = make_image_bytes(seed=101)
    image_b = make_image_bytes(seed=202)

    with httpx.Client(timeout=30) as client:
        r1 = send_request(client, image_a, "pytorch")
        print(f"  1. pytorch, fresh image  -> cache_hit={r1['cache_hit']}, "
              f"latency={r1['total_latency_ms']}ms")

        r2 = send_request(client, image_b, "onnx")
        print(f"  2. onnx, fresh image     -> cache_hit={r2['cache_hit']}, "
              f"latency={r2['total_latency_ms']}ms")

        r3 = send_request(client, image_a, "pytorch")
        print(f"  3. pytorch, repeat image -> cache_hit={r3['cache_hit']}, "
              f"latency={r3['total_latency_ms']}ms")

    print("\nQuerying request_log directly in PostgreSQL for the most recent rows...\n")
    rows = asyncio.run(fetch_recent_rows(limit=5))

    if not rows:
        print("No rows found - is the service running with a reachable PostgreSQL instance?")
        return

    header = f"{'id':>4}  {'created_at':<26}  {'backend':<11}  {'cache_hit':<9}  {'batch_size':<10}  {'class_id':<8}  latency_ms"
    print(header)
    print("-" * len(header))
    for row in rows:
        batch_size_str = str(row["batch_size"]) if row["batch_size"] is not None else "NULL"
        print(f"{row['id']:>4}  {str(row['created_at']):<26}  {row['backend']:<11}  "
              f"{str(row['cache_hit']):<9}  {batch_size_str:<10}  {row['predicted_class_id']:<8}  "
              f"{row['total_latency_ms']:.2f}")

    print(f"\n{len(rows)} row(s) confirmed in PostgreSQL - request logging is working.")


if __name__ == "__main__":
    main()
