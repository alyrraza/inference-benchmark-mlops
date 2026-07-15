"""
Eager-mode PyTorch inference backend - the baseline, no optimization applied.
"""

import numpy as np
import torch
from transformers import ViTForImageClassification

from app import config
from app.inference.base import InferenceBackend


class PyTorchBackend(InferenceBackend):
    name = "pytorch"

    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained(config.MODEL_NAME)
        self.model.eval()  # disables dropout - required for deterministic inference

    def predict(self, batch: torch.Tensor) -> np.ndarray:
        # no_grad() skips building the autograd graph, which we don't need
        # for inference and which would otherwise waste memory and time
        # tracking gradients we're never going to compute.
        with torch.no_grad():
            output = self.model(batch)
        return output.logits.numpy()
