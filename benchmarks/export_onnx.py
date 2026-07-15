"""
Exports google/vit-base-patch16-224 to ONNX for the CPU benchmark.

WHY THIS SCRIPT EXISTS INSTEAD OF REUSING kaggle/vit_base.onnx:
The Kaggle GPU notebook already exported this model to ONNX, but Kaggle's
newer dynamo-based exporter split the file into a small graph-definition
file (vit_base.onnx, ~100KB) plus a separate external-data file holding the
actual weight tensors (vit_base.onnx.data). Only the small graph file was
saved locally - the weight data file never made it off Kaggle. ONNX Runtime
needs both to load the model, so that file can't be used here as-is.

This is not "redoing the GPU benchmark work" - the actual GPU benchmark
numbers in kaggle/results/benchmark_results.json are untouched and are not
being reproduced. This script only re-creates a loadable ONNX file, using
the same export settings as the Kaggle notebook (opset 17, dynamic batch
axis), so the CPU benchmark has something to load.
"""

import torch
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_PATH = "benchmarks/vit_base_cpu.onnx"


def export():
    print(f"Loading {MODEL_NAME} from Hugging Face...")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("Exporting to ONNX (opset 17, dynamic batch axis)...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"Saved ONNX model to {OUTPUT_PATH}")


if __name__ == "__main__":
    export()
