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
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _fake_image_bytes() -> bytes:
    array = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_pytorch_backend_loaded():
    # `with TestClient(app)` triggers FastAPI's lifespan handler - startup
    # code (loading backends, starting the BatchWorker) actually runs here,
    # the same as a real `uvicorn app.main:app` launch would.
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "pytorch" in body["backends_loaded"]


def test_predict_returns_a_prediction():
    with TestClient(app) as client:
        files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
        response = client.post("/predict", files=files, params={"backend": "pytorch"})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class_id" in data
        assert "predicted_label" in data
        assert data["backend"] == "pytorch"
        assert isinstance(data["cache_hit"], bool)


def test_predict_rejects_unknown_backend():
    with TestClient(app) as client:
        files = {"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
        response = client.post("/predict", files=files, params={"backend": "not_a_real_backend"})
        assert response.status_code == 400
