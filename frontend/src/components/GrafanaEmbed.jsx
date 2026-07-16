const GRAFANA_BASE = import.meta.env.VITE_GRAFANA_URL || "http://127.0.0.1:3000";

// The bare `kiosk` param (no value) hides Grafana's own navbar,
// breadcrumbs, search, and sign-in chrome, leaving just the panels and
// time-range picker - the standard way to embed a Grafana dashboard
// inside another page without it looking like "a website inside a
// website". This was verified visually, not assumed: `kiosk=tv` (an
// older/different Grafana kiosk variant) did NOT hide the chrome on this
// Grafana version - only the bare `kiosk` param did. refresh=5s matches
// this project's own scrape interval (prometheus/prometheus.yml), so the
// embedded view updates at the same cadence new data can actually arrive.
const DASHBOARD_URL =
  `${GRAFANA_BASE}/d/inferbench-main/inferbench` +
  `?orgId=1&kiosk&refresh=5s&from=now-15m&to=now`;

export default function GrafanaEmbed() {
  return (
    <div className="card grafana-card">
      <div className="grafana-card__header">
        <h2 className="card__title" style={{ margin: 0 }}>
          Live Metrics (Grafana)
        </h2>
        <a href={DASHBOARD_URL.replace("&kiosk", "")} target="_blank" rel="noreferrer">
          Open in Grafana ↗
        </a>
      </div>
      <div className="grafana-frame-wrap">
        <iframe src={DASHBOARD_URL} title="InferBench Grafana dashboard" />
      </div>
    </div>
  );
}
