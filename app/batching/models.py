"""
The data structure that travels through the request queue.

One QueueItem is created per incoming HTTP request, put onto the shared
asyncio.Queue, and later picked up by the BatchWorker.
"""

import asyncio
from dataclasses import dataclass

import torch


@dataclass
class QueueItem:
    image: torch.Tensor  # preprocessed, shape [3, 224, 224] - one image
    backend: str  # which model this request wants ("pytorch"/"onnx"/"torchscript")
    future: asyncio.Future  # how the result gets back to the waiting request handler
    enqueued_at: float  # time.perf_counter() when this item was queued
    batch_size: int = 0  # filled in by the worker: how many requests shared this item's forward pass
