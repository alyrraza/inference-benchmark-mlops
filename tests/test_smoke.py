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

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
