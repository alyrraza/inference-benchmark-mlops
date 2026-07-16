import { useEffect, useState } from "react";
import Header from "./components/Header";
import TechStack from "./components/TechStack";
import PredictPanel from "./components/PredictPanel";
import StatsStrip from "./components/StatsStrip";
import GrafanaEmbed from "./components/GrafanaEmbed";
import { getHealth } from "./api";
import "./App.css";

function App() {
  const [healthy, setHealthy] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    getHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  return (
    <div className="app">
      <Header healthy={healthy} />
      <TechStack />

      <div className="layout">
        <div>
          <PredictPanel onPredicted={() => setRefreshKey((k) => k + 1)} />
          <StatsStrip refreshKey={refreshKey} />
        </div>
        <GrafanaEmbed />
      </div>

      <p className="footer">
        Local demo control panel - separate from the Phase 7 Gradio deployment.
        See docs/concepts/05c_demo_frontend.md.
      </p>
    </div>
  );
}

export default App;
