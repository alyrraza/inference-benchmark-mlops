"""
Turns raw uploaded image bytes into the normalized tensor shape ViT expects.

Uses the same ViTImageProcessor Hugging Face ships with the model, so this
matches the exact resize/normalize steps used when the model was trained
and when it was benchmarked in Phase 1 - a different resize or normalization
would silently shift every prediction.
"""

import io
from functools import lru_cache

import torch
from PIL import Image
from transformers import ViTImageProcessor

from app import config


@lru_cache(maxsize=1)
def get_processor() -> ViTImageProcessor:
    # Cached so we parse the processor's config.json once, not on every
    # request - it never changes at runtime.
    return ViTImageProcessor.from_pretrained(config.MODEL_NAME)


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    image_bytes: raw file contents from an uploaded image (jpg/png/etc.)
    returns: float32 tensor, shape [3, 224, 224] - a single preprocessed image,
             not yet batched with anything else.
    """
    # convert("RGB") guards against grayscale or RGBA uploads, which would
    # otherwise produce a tensor with the wrong number of channels and crash
    # the model's first conv layer.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processor = get_processor()
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"][0]
