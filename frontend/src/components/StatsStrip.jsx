import { useEffect, useState } from "react";
import { getHealth, getRequestTotals } from "../api";

const POLL_INTERVAL_MS = 4000;

export default function StatsStrip({ refreshKey }) {
  const [totals, setTotals] = useState(null);
  const [backendCount, setBackendCount] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const [health, requestTotals] = await Promise.all([getHealth(), getRequestTotals()]);
        if (cancelled) return;
        setBackendCount(health.backends_loaded.length);
        setTotals(requestTotals);
      } catch {
        // Stats are a nice-to-have overlay, not core functionality - a
        // failed poll just leaves the last known values on screen rather
        // than showing an error banner over a demo dashboard.
      }
    }

    refresh();
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // refreshKey changes right after a prediction completes, so the
    // strip updates immediately instead of waiting for the next poll tick.
  }, [refreshKey]);

  const hitRateText =
    totals?.hitRate == null ? "—" : `${(totals.hitRate * 100).toFixed(1)}%`;

  return (
    <div className="stats-strip">
      <div className="stat-tile">
        <p className="stat-tile__label">Total Requests</p>
        <p className="stat-tile__value">{totals ? Math.round(totals.total) : "—"}</p>
      </div>
      <div className="stat-tile">
        <p className="stat-tile__label">Cache Hit Rate</p>
        <p className="stat-tile__value">{hitRateText}</p>
      </div>
      <div className="stat-tile">
        <p className="stat-tile__label">Active Backends</p>
        <p className="stat-tile__value">{backendCount ?? "—"}</p>
      </div>
    </div>
  );
}
