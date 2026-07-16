// Thin fetch wrappers around the InferBench FastAPI service. No API
// client library - this is a small enough surface (four endpoints) that
// a dedicated library would be more code to configure than to just call
// fetch() directly.

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function predict(file, backend) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict?backend=${backend}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }

  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  return response.json();
}

export async function getCacheStats() {
  const response = await fetch(`${API_BASE}/cache/stats`);
  if (!response.ok) throw new Error(`Cache stats failed (${response.status})`);
  return response.json();
}

// /metrics returns Prometheus's plain-text exposition format, not JSON -
// this is a deliberately minimal parser, not a general Prometheus text
// format parser. It only extracts what the stats strip needs: the sum of
// inferbench_requests_total across every backend/cache_hit label
// combination, split by cache_hit value. A real Prometheus client library
// exists for this in Python (see app/metrics.py); nothing equivalent is
// needed here since this is read-only, throwaway parsing for a demo UI,
// not something anything downstream depends on being fully correct.
export async function getRequestTotals() {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) throw new Error(`Metrics fetch failed (${response.status})`);
  const text = await response.text();

  let hits = 0;
  let misses = 0;

  for (const line of text.split("\n")) {
    if (!line.startsWith("inferbench_requests_total{")) continue;
    const valueMatch = line.match(/}\s+([0-9.eE+-]+)\s*$/);
    if (!valueMatch) continue;
    const value = parseFloat(valueMatch[1]);
    if (line.includes('cache_hit="true"')) {
      hits += value;
    } else if (line.includes('cache_hit="false"')) {
      misses += value;
    }
  }

  const total = hits + misses;
  return {
    total,
    hits,
    misses,
    hitRate: total > 0 ? hits / total : null,
  };
}
