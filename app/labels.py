"""
Maps the model's raw output index (e.g. 285) to a human-readable ImageNet
class name (e.g. "Egyptian cat"). Loaded once and cached - this mapping is
identical across all three backends since they all share the same weights,
so it doesn't belong inside any one InferenceBackend implementation.
"""

from functools import lru_cache

from transformers import ViTConfig

from app import config


@lru_cache(maxsize=1)
def get_id2label() -> dict[int, str]:
    model_config = ViTConfig.from_pretrained(config.MODEL_NAME)
    # Hugging Face configs sometimes load this mapping with string keys
    # (JSON object keys are always strings) - normalize to int here so
    # callers can look up with the int class id numpy's argmax gives them.
    return {int(k): v for k, v in model_config.id2label.items()}


def label_for(class_id: int) -> str:
    return get_id2label().get(class_id, f"unknown_class_{class_id}")
