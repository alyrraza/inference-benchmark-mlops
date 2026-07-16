# InferBench API service image.
#
# Model artifacts (the ONNX and TorchScript exports from Phase 1) are
# built INSIDE this image, not copied in from the host. Those files are
# gitignored (large, regenerable binaries - see .gitignore), so a fresh
# `git clone` has no local copies to copy in even if we wanted to. Baking
# the export step into the build means `docker build` alone produces a
# fully self-contained image - no separate manual export step required
# before `docker-compose up`, matching this phase's "one command, no
# manual steps" goal. The cost is a slower build (downloads ViT-Base's
# weights from Hugging Face, ~350MB, plus export time) - a one-time cost
# per build, not per container start.
FROM python:3.11-slim

WORKDIR /app

# curl is needed for the HEALTHCHECK below - not part of the base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision from the CPU-only wheel index first (same reasoning
# as every local setup in this project - see requirements.txt's own
# comment), then everything else from PyPI via requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Only what's needed to serve the API and export the model artifacts -
# not scripts/, not kaggle/, not the benchmark result JSONs. Keeps the
# image focused on "run the service," not "reproduce the whole project."
COPY app/ ./app/
COPY benchmarks/export_torchscript.py benchmarks/export_onnx.py ./benchmarks/

# No D:-drive-style path juggling needed here, unlike every local Windows
# setup in this project - inside a container, the whole filesystem is
# already isolated and disposable, so there's no "don't waste C: drive
# space" concern to work around. HF_HOME just needs to point somewhere
# writable inside the image.
ENV HF_HOME=/app/.hf-cache
RUN python benchmarks/export_torchscript.py && python benchmarks/export_onnx.py

EXPOSE 8000

# start-period is generous (90s) because startup genuinely takes real
# time here: loading three separate model backends (Phase 2's
# load_all_backends()) before the process can answer anything, not a
# quick "process started" check.
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
