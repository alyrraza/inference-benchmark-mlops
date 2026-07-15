"""
ONNX Runtime inference backend - loads the graph exported in Phase 1
(benchmarks/export_onnx.py), running on CPUExecutionProvider.
"""

import numpy as np
import onnxruntime as ort
import torch

from app import config
from app.inference.base import InferenceBackend


class ONNXBackend(InferenceBackend):
    name = "onnx"

    def __init__(self):
        session_options = ort.SessionOptions()
        # Pinned to the same thread budget as PyTorch (config.TORCH_NUM_THREADS)
        # so a backend comparison isn't confounded by one engine quietly
        # using more CPU threads than another - see config.py's comment on
        # TORCH_NUM_THREADS for why this matters.
        session_options.intra_op_num_threads = config.TORCH_NUM_THREADS
        self.session = ort.InferenceSession(
            config.ONNX_MODEL_PATH,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, batch: torch.Tensor) -> np.ndarray:
        input_np = batch.numpy()
        outputs = self.session.run(None, {"pixel_values": input_np})
        return outputs[0]
