"""
TorchScript inference backend - loads the traced graph exported in Phase 1
(benchmarks/export_torchscript.py) instead of re-tracing the model here.
"""

import numpy as np
import torch

from app import config
from app.inference.base import InferenceBackend


class TorchScriptBackend(InferenceBackend):
    name = "torchscript"

    def __init__(self):
        self.model = torch.jit.load(config.TORCHSCRIPT_MODEL_PATH)
        self.model.eval()

    def predict(self, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            output = self.model(batch)

        # torch.jit.trace flattens Hugging Face's ModelOutput object into a
        # plain dict during tracing - we hit this exact surprise in Phase 1's
        # accuracy-validation code (benchmarks/run_cpu_benchmark.py). Only
        # the eager model still returns an object with a .logits attribute.
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        return logits.numpy()
