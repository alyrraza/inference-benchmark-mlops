"""
The common interface every inference backend implements.

WHY THIS EXISTS:
The batch worker (app/batching/worker.py) needs to call "run this batch of
images through the model" without caring whether the model underneath is
eager PyTorch, an ONNX Runtime session, or a TorchScript graph - those three
have completely different Python APIs (model(tensor) vs session.run(...) vs
model(tensor) but with a different return type). This abstract base class
is the contract that makes them interchangeable from the worker's point of
view. This is the Strategy pattern: swap the algorithm (inference engine)
without changing the code that uses it.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, batch: torch.Tensor) -> np.ndarray:
        """
        Runs a forward pass on a batch of preprocessed images.

        batch: float32 tensor, shape [N, 3, 224, 224] - N images already
               resized and normalized by app/preprocessing.py.
        returns: float32 numpy array, shape [N, num_classes] - raw logits,
                 one row per input image, in the same order as the input.
        """
        raise NotImplementedError
