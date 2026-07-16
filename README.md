# InferBench

A self-serve model-inference optimization and observability platform. Upload
a model, get back a benchmarked, monitored inference endpoint, with real
before/after latency numbers across PyTorch, ONNX Runtime, and
TensorRT/TorchScript.

This build serves `google/vit-base-patch16-224` (ViT-Base), benchmarked on
both GPU (Kaggle T4) and CPU (local), and served through a FastAPI backend
with a dynamic batching layer built from scratch.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU-black)
![Redis](https://img.shields.io/badge/Redis-caching-DC382D)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
[![CI](https://github.com/alyrraza/inference-benchmark-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/alyrraza/inference-benchmark-mlops/actions/workflows/ci.yml)

## Demo video

[![Watch the InferBench demo](screenshots/frontend_dashboard.png)](screenshots/inferbench_demo.mp4)

Click the image above (or [this link](screenshots/inferbench_demo.mp4)) to
watch a real prediction go through the stack, with the Grafana dashboard
updating live as it happens. Recorded using the local demo control panel
described further down - see "Local demo frontend" for what that is and
how it differs from the Phase 7 deployment.

## Architecture

```mermaid
graph TB
    U["User / Load Test Script"]

    subgraph Stack["Local Stack (CPU-only)"]
        API["FastAPI Service"]
        RQ["Request Queue"]
        BW["Batch Worker (10ms window)"]
        Redis[("Redis\ncache + queue backing")]
        PG[("PostgreSQL\nbenchmark/metadata store")]
        Prom["Prometheus"]
        Graf["Grafana"]
        PT["PyTorch Baseline"]
        ONNX["ONNX Runtime (CPU)"]
        TS["TorchScript"]
    end

    subgraph Kaggle["Kaggle Notebook (T4 GPU, benchmarking only)"]
        ONNXCUDA["ONNX Runtime (CUDA)"]
        TRT["TensorRT (FP16)"]
    end

    U -->|"POST /predict (image)"| API
    API --> RQ
    BW --> RQ
    BW --> PT
    BW --> ONNX
    BW --> TS
    API --> Redis
    API --> PG
    API -->|"/metrics"| Prom
    Prom --> Graf
    Graf -->|"dashboard"| U
```

The GPU benchmark (Kaggle) is offline, one-time work, its results are
already produced and just get referenced by the deployed CPU service, not
reproduced.

## Build status

| Phase | What | Status |
|---|---|---|
| 1 | CPU benchmarking (PyTorch / ONNX Runtime / TorchScript, batch 1-16) | Done |
| 2 | FastAPI service + dynamic batching layer built from scratch | Done |
| 3 | Redis response caching + GitHub Actions CI | Done |
| 4 | PostgreSQL benchmark/metadata store | Done |
| 5 | Prometheus + Grafana observability | Done |
| 6 | Docker Compose orchestration | Not started |
| 7 | Gradio demo (Hugging Face Spaces) | Not started |
| 8 | Final README + deployment | Not started |

## Results so far

**GPU (Kaggle, Tesla T4)** - see `kaggle/results/benchmark_results.json`:

| Batch size | PyTorch | ONNX Runtime (CUDA) | TensorRT FP16 | TRT speedup |
|---|---|---|---|---|
| 1  | 14.9ms  | 15.4ms  | 5.1ms  | 2.9x |
| 4  | 47.1ms  | 46.9ms  | 8.5ms  | 5.6x |
| 8  | 80.7ms  | 94.8ms  | 15.2ms | 5.3x |
| 16 | 166.9ms | 189.7ms | 32.5ms | 5.1x |

**CPU (local, this repo)** - see `benchmarks/results/cpu_benchmark_results.json`:

| Batch size | PyTorch (eager) | ONNX Runtime (CPU) | TorchScript |
|---|---|---|---|
| 1  | 263.2ms  | 280.3ms  | 306.7ms  |
| 4  | 945.2ms  | 1146.0ms | 1055.1ms |
| 8  | 1830.9ms | 2159.0ms | 2082.9ms |
| 16 | 3783.0ms | 4406.9ms | 4148.9ms |

Worth noting honestly: on this CPU, plain eager PyTorch was actually
faster than both ONNX Runtime and TorchScript at every batch size. That's
a real, measured result, not an error, format conversion is not an
automatic speedup, it depends on the hardware. TensorRT's GPU-specific
kernel fusion is what produced the large GPU-side win, there's no direct
CPU equivalent in this comparison.

## Observability

Live Grafana dashboard (provisioned from `grafana/dashboards/inferbench.json`,
not clicked together manually), showing real traffic from a concurrent
load test - latency percentiles, throughput by backend and cache hit/miss,
and the batch size distribution the dynamic batching worker actually formed:

![Grafana dashboard showing InferBench metrics](screenshots/grafana_dashboard.png)

Prometheus confirming it's actually scraping the service's `/metrics`
endpoint (not just configured to, genuinely up and current):

![Prometheus target health showing the inferbench job up](screenshots/prometheus_targets.png)

## Tech stack

- **React (Vite)** - local-only demo control panel for recording videos,
  see "Local demo frontend" below - not part of the locked backend
  architecture and not deployed anywhere
- **FastAPI** - REST API layer
- **PyTorch / ONNX Runtime / TorchScript** - three interchangeable CPU
  inference backends behind one common interface
- **A hand-built dynamic batching layer** - `asyncio.Queue` +
  `asyncio.Future`, no batching library
- **Redis** - response cache, keyed on a hash of the image bytes + backend
- **PostgreSQL** - `request_log` table storing backend, cache hit/miss,
  batch size, predicted class, latency, and timestamp for every request
- **Prometheus + Grafana** - latency histogram (p50/p95/p99), request
  counter, and batch-size histogram, scraped from `/metrics` every 5s and
  visualized on a dashboard provisioned entirely from version-controlled
  config
- **Docker Compose** - multi-service orchestration (Phase 6)
- **GitHub Actions** - CI on every push/PR: installs dependencies, runs
  the pytest smoke suite against real Redis and PostgreSQL service
  containers

## Running locally

Requires Python 3.11+. All dependencies install into a local virtual
environment inside the project folder.

```powershell
cd "D:\MLOps\Infer Bench"
python -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A local Redis instance is needed for caching (optional - the service
runs fine without one, every request just becomes a cache miss):

```powershell
winget install Redis.Redis --accept-package-agreements --accept-source-agreements --silent
```

A local PostgreSQL instance is needed for request logging (also optional -
the service runs fine without one, it just stops recording history).
This project uses EDB's portable binaries zip instead of the installer,
so nothing gets registered as a Windows service and no admin rights are
needed:

```powershell
Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-16.4-1-windows-x64-binaries.zip" -OutFile "postgresql-binaries.zip"
Expand-Archive -Path "postgresql-binaries.zip" -DestinationPath ".postgres\" -Force
Remove-Item "postgresql-binaries.zip"
.\.postgres\pgsql\bin\initdb.exe -D ".\.postgres-data" -U postgres -A trust --encoding=UTF8
.\.postgres\pgsql\bin\pg_ctl.exe -D ".\.postgres-data" -o "-p 5433" -l ".\.postgres-data\logfile.log" start
.\.postgres\pgsql\bin\createdb.exe -U postgres -p 5433 inferbench
```

Export the model artifacts once (only needed the first time):

```powershell
.venv\Scripts\python.exe benchmarks\export_torchscript.py
.venv\Scripts\python.exe benchmarks\export_onnx.py
```

Run the tests:

```powershell
$env:HF_HOME = "D:\MLOps\Infer Bench\.hf-cache"
.venv\Scripts\python.exe -m pytest tests\ -v
```

Start the service:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Test it (in a second terminal):

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s -X POST "http://127.0.0.1:8000/predict?backend=pytorch" -F "file=@your_image.jpg;type=image/jpeg"
curl.exe -s http://127.0.0.1:8000/cache/stats
```

Run the concurrent load test to see the batching layer group requests:

```powershell
.venv\Scripts\python.exe scripts\load_test.py
```

Run the cache miss/hit demonstration:

```powershell
.venv\Scripts\python.exe scripts\verify_cache.py
```

Run the database logging demonstration:

```powershell
.venv\Scripts\python.exe scripts\verify_db_logging.py
```

Prometheus and Grafana (also optional, also portable binaries, no admin
rights or Windows services - see
`docs/concepts/05b_phase5_walkthrough.md` for the full setup including a
Grafana startup gotcha worth knowing about):

```powershell
# Prometheus
Invoke-WebRequest -Uri "https://github.com/prometheus/prometheus/releases/download/v3.13.1/prometheus-3.13.1.windows-amd64.zip" -OutFile "prometheus.zip"
Expand-Archive -Path "prometheus.zip" -DestinationPath ".prometheus\" -Force
.\.prometheus\prometheus-3.13.1.windows-amd64\prometheus.exe --config.file="prometheus\prometheus.yml" --storage.tsdb.path=".prometheus-data" --web.listen-address=127.0.0.1:9090

# Grafana (separate terminal)
Invoke-WebRequest -Uri "https://dl.grafana.com/oss/release/grafana-13.1.0.windows-amd64.zip" -OutFile "grafana.zip"
Expand-Archive -Path "grafana.zip" -DestinationPath ".grafana\" -Force
$env:GF_PATHS_PROVISIONING = "$PWD\grafana\provisioning"
$env:GF_PATHS_DATA = "$PWD\.grafana-data"
.\.grafana\grafana-13.1.0\bin\grafana.exe server --homepath=".grafana\grafana-13.1.0"
```

Open `http://127.0.0.1:3000` (login `admin`/`admin`) - the InferBench
dashboard is already provisioned, no manual setup needed.

## Local demo frontend

`frontend/` is a small React (Vite) single-page app - a polished local
"mission control" panel for recording demo videos, **not** part of the
locked backend architecture in `docs/architecture_diagram.puml`, and
**not** the same thing as Phase 7's Gradio deployment. See
`docs/concepts/05c_demo_frontend.md` for the full explanation of why
both exist: this one only ever runs on `localhost` and is never
deployed anywhere; Phase 7's Gradio app is the actual public, permanent
demo link for recruiters.

It has an upload-and-predict panel (drag-and-drop, backend selector,
animated results), a live stats strip, and the same Grafana dashboard
embedded via iframe so metrics update in real time as you make
predictions.

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Requires the FastAPI service to be running
(with CORS enabled, which it is by default - see `app/config.py`'s
`CORS_ALLOWED_ORIGINS`) and, for the embedded dashboard panel to show
live data, Grafana running with `GF_SECURITY_ALLOW_EMBEDDING=true` (not
needed for the rest of this project, only for this iframe embed - see
the walkthrough doc for the exact launch command).

## Docker

Not set up yet - coming in Phase 6, which brings every one of these
standalone local processes (Redis, PostgreSQL, Prometheus, Grafana) under
one `docker-compose up`.

## Demo

Not deployed publicly yet - coming in Phase 7 (Gradio on Hugging Face
Spaces), which will be the permanent, public demo link. The React
frontend described above is a separate, local-only tool for recording
demo videos, not a deployment target.
