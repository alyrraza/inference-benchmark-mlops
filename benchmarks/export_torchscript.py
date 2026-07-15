"""
Exports google/vit-base-patch16-224 to TorchScript.

WHY THIS FILE EXISTS:
PyTorch's default mode ("eager mode") re-interprets your Python model code
every single time you call it. TorchScript runs a tracer once, records the
actual sequence of tensor operations the model performs, and saves that as a
static graph. That static graph can then run without going back through the
Python interpreter for every layer, which is why it's faster on CPU.

We do this export once, save the result to disk, and every benchmark run
after this just loads the saved .pt file instead of re-tracing the model.
"""

import torch
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_PATH = "benchmarks/vit_base_torchscript.pt"


def export():
    print(f"Loading {MODEL_NAME} from Hugging Face...")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()  # disables dropout/batchnorm training behavior - required before tracing

    # A dummy input with the exact shape the real model expects:
    # (batch_size=1, channels=3, height=224, width=224)
    # We trace with batch_size=1; TorchScript's traced graph still works for
    # other batch sizes because the batch dimension isn't hardcoded into the
    # operations themselves, only the data shape is.
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Tracing model into TorchScript graph...")
    # torch.jit.trace runs the model once with dummy_input and records every
    # tensor operation that happened along the way. strict=False lets it
    # ignore some Python control-flow warnings that don't affect ViT's forward pass.
    traced_model = torch.jit.trace(model, dummy_input, strict=False)

    traced_model.save(OUTPUT_PATH)
    print(f"Saved TorchScript model to {OUTPUT_PATH}")


if __name__ == "__main__":
    export()
