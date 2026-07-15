"""
Builds every inference backend once at startup.

WHY THIS EXISTS:
Loading a model (reading weights off disk, constructing the ONNX Runtime
session) takes real time - hundreds of milliseconds to a couple seconds.
Doing that inside a request handler would mean every request pays that cost,
which defeats the entire point of a "fast" inference service. Instead,
load_all_backends() runs exactly once, when the FastAPI app starts up (see
the lifespan handler in app/main.py), and the resulting objects are reused
for every request for the lifetime of the process.
"""

from app.inference.base import InferenceBackend
from app.inference.onnx_backend import ONNXBackend
from app.inference.pytorch_backend import PyTorchBackend
from app.inference.torchscript_backend import TorchScriptBackend

_BACKEND_CLASSES = {
    "pytorch": PyTorchBackend,
    "torchscript": TorchScriptBackend,
    "onnx": ONNXBackend,
}


def load_all_backends() -> dict[str, InferenceBackend]:
    backends = {}
    for name, backend_cls in _BACKEND_CLASSES.items():
        print(f"Loading backend '{name}'...")
        backends[name] = backend_cls()
    return backends
