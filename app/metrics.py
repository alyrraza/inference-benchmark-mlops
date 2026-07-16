"""
Prometheus metrics for the InferBench service, exposed at GET /metrics in
the plain-text format Prometheus itself scrapes.

Three metrics, matching this phase's scope:
- REQUEST_LATENCY_MS: a histogram of end-to-end /predict latency, labeled
  by backend and cache_hit - Prometheus's histogram_quantile() function
  can derive p50/p95/p99 from this after the fact, in a query, without
  this service ever computing a percentile itself.
- REQUEST_COUNT: a counter of total requests served, same labels - the
  rate() of this over time is throughput (requests/second).
- BATCH_SIZE: a histogram of the actual batch size the BatchWorker used
  for each batch it ran, recorded once per batch (not once per request
  in that batch) - this is what "batch size distribution" means: how
  many batches were size 1 vs size 16, not how many requests happened to
  be in a large batch.

Unlike app/cache.py and app/db.py, none of this needs error handling for
an unreachable dependency - recording a metric is a pure, in-memory
Python operation with no network call involved. Prometheus is the one
that reaches out to *this* service (by scraping /metrics on its own
schedule), not the other way around - see
docs/concepts/05_prometheus_grafana.md for why that direction matters.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_LATENCY_MS = Histogram(
    "inferbench_request_latency_ms",
    "End-to-end /predict request latency in milliseconds",
    ["backend", "cache_hit"],
    # Spans the real range this project has actually measured: cache hits
    # under 1ms (Phase 3 measured 0.84ms), cache misses from a couple
    # hundred ms up to several seconds at batch size 16 (Phase 1 measured
    # up to ~4.4 seconds for ONNX Runtime CPU at batch 16).
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

REQUEST_COUNT = Counter(
    "inferbench_requests_total",
    "Total number of /predict requests served",
    ["backend", "cache_hit"],
)

BATCH_SIZE = Histogram(
    "inferbench_batch_size",
    "Number of requests grouped into each batch the worker actually ran",
    ["backend"],
    buckets=(1, 2, 4, 8, 16),
)


def record_request(backend: str, cache_hit: bool, latency_ms: float) -> None:
    # Prometheus label values are always strings - "true"/"false" here,
    # not Python's True/False, since labels().observe() needs a string key.
    cache_hit_label = "true" if cache_hit else "false"
    REQUEST_LATENCY_MS.labels(backend=backend, cache_hit=cache_hit_label).observe(latency_ms)
    REQUEST_COUNT.labels(backend=backend, cache_hit=cache_hit_label).inc()


def record_batch(backend: str, batch_size: int) -> None:
    BATCH_SIZE.labels(backend=backend).observe(batch_size)


def latest_metrics_text() -> bytes:
    return generate_latest()
