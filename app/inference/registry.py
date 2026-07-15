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
    """
    Loads every backend independently - one backend failing to load (a
    missing model file, a missing optional dependency) does not prevent
    the others from starting. This matters for CI specifically: the ONNX
    and TorchScript model files are build-time artifacts, not committed to
    git (see .gitignore), so a fresh checkout only has the PyTorch backend
    available (it downloads its weights straight from Hugging Face). The
    service should still start and serve traffic with whatever backends it
    could load, rather than refusing to start at all because two out of
    three happened to be unavailable.
    """
    backends: dict[str, InferenceBackend] = {}
    for name, backend_cls in _BACKEND_CLASSES.items():
        print(f"Loading backend '{name}'...")
        try:
            backends[name] = backend_cls()
        except Exception as exc:
            print(f"[registry] backend '{name}' failed to load, skipping: {exc}")

    if not backends:
        raise RuntimeError("No inference backends could be loaded - check model file paths and dependencies.")

    return backends
