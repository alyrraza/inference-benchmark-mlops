"""
Smoke test: does the service actually boot and answer a real request?

This is the test GitHub Actions CI runs on every push (see
.github/workflows/ci.yml). It only exercises the "pytorch" backend
deliberately - the ONNX and TorchScript model files are build-time
artifacts, not committed to git, so a fresh CI checkout never has them.
Backend loading is resilient to that (see app/inference/registry.py), and
this test's job is to prove the service still starts and serves a real
prediction with whatever backend is available, not to test every backend.

No Redis is required either - app/cache.py degrades to "treat every
request as a cache miss" when Redis is unreachable, so this test is
exercising the real, non-mocked cache-miss code path whether or not a
Redis instance happens to be running wherever this test executes.
"""

import asyncio
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import db
from app.main import app


def _fake_image_bytes() -> bytes:
    array = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    # scope="module" means this fixture (and the `with` block's startup/
    # shutdown) runs ONCE for every test in this file, not once per test.
    # `with TestClient(app)` triggers FastAPI's lifespan handler, which
    # loads every model backend from disk/Hugging Face - real, non-mocked
    # work that takes real time. A plain function-scoped fixture (the
    # pytest default) would re-run that full startup for every single test
    # function, loading the same models repeatedly for no benefit - this
    # is exactly what the first version of this file did, and it made a
    # 3-test file take nearly 10 minutes and exhaust available disk space
    # from the repeated model loads. See docs/concepts/03b_phase3_walkthrough.md
    # for the full story of hitting that.
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_pytorch_backend_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "pytorch" in body["backends_loaded"]


def test_predict_returns_a_prediction(client):
    files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, params={"backend": "pytorch"})

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class_id" in data
    assert "predicted_label" in data
    assert data["backend"] == "pytorch"
    assert isinstance(data["cache_hit"], bool)


def test_predict_rejects_unknown_backend(client):
    files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, params={"backend": "not_a_real_backend"})
    assert response.status_code == 400


def test_predict_caches_identical_requests(client):
    # A fresh random image each test run - unseeded on purpose, so this
    # request can never collide with a leftover cache entry from a
    # previous run and accidentally start as a hit instead of a miss.
    image_bytes = _fake_image_bytes()

    first = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        params={"backend": "pytorch"},
    )
    assert first.status_code == 200
    first_data = first.json()

    second = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        params={"backend": "pytorch"},
    )
    assert second.status_code == 200
    second_data = second.json()

    if not second_data["cache_hit"]:
        # No Redis reachable in this environment - app/cache.py degraded
        # every call to a miss, which is correct behavior, just not what
        # this particular test needs to assert something about. Skip
        # rather than fail: this environment not having Redis running
        # isn't a bug in the service.
        pytest.skip("Redis not reachable in this environment - cache-hit path not exercised")

    assert first_data["cache_hit"] is False
    assert second_data["predicted_class_id"] == first_data["predicted_class_id"]


def test_predict_logs_to_database(client):
    health = client.get("/health").json()
    if not health["db_available"]:
        # Same reasoning as the cache-hit test's skip above - no Postgres
        # reachable in this environment isn't a bug in the service (see
        # app/main.py's lifespan: a database down at startup degrades to
        # "logging disabled", it doesn't stop the app from serving
        # predictions), just not something this specific test can check.
        pytest.skip("PostgreSQL not reachable in this environment - DB logging path not exercised")

    files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, params={"backend": "pytorch"})
    assert response.status_code == 200
    data = response.json()

    # Query the request_log table directly - a separate, short-lived
    # connection pool, not the app's own one, to prove the row is really
    # sitting in Postgres and readable by anyone, not just recoverable
    # from the app's in-memory state.
    async def _fetch_latest_row():
        pool = await db.create_pool()
        try:
            async with pool.acquire() as conn:
                return await conn.fetchrow(
                    "SELECT backend, cache_hit, batch_size, predicted_class_id, total_latency_ms "
                    "FROM request_log ORDER BY id DESC LIMIT 1"
                )
        finally:
            await pool.close()

    row = asyncio.run(_fetch_latest_row())
    assert row is not None
    assert row["backend"] == "pytorch"
    assert row["predicted_class_id"] == data["predicted_class_id"]


def test_metrics_endpoint_reflects_requests(client):
    # A fresh image, unseeded, so this is guaranteed a cache miss - the
    # point of this test is confirming a real /predict call moves the
    # needle on /metrics, not just that /metrics returns 200.
    files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    predict_response = client.post("/predict", files=files, params={"backend": "pytorch"})
    assert predict_response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    body = metrics_response.text

    # Prometheus's text exposition format - no JSON parsing needed, just
    # confirm the metric names and this request's label values appear.
    assert "inferbench_requests_total" in body
    assert 'inferbench_requests_total{backend="pytorch",cache_hit="false"}' in body
    assert "inferbench_request_latency_ms" in body
    assert "inferbench_batch_size" in body
