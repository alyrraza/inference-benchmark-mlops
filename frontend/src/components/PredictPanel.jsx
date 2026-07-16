import { useState, useRef } from "react";
import { predict } from "../api";

const BACKENDS = ["pytorch", "onnx", "torchscript"];

export default function PredictPanel({ onPredicted }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [backend, setBackend] = useState("pytorch");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  function selectFile(selected) {
    if (!selected) return;
    setFile(selected);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function handlePredict() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predict(file, backend);
      setResult(data);
      onPredicted?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2 className="card__title">Predict</h2>

      <div
        className={`dropzone ${dragActive ? "dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
        <div className="dropzone__icon">📷</div>
        <div className="dropzone__text">
          <strong>{file ? file.name : "Drop an image, or click to browse"}</strong>
          <br />
          JPEG or PNG
        </div>
      </div>

      {previewUrl && (
        <div className="preview">
          <img src={previewUrl} alt="Selected upload preview" />
        </div>
      )}

      <div className="backend-select">
        <span className="backend-select__label">Backend</span>
        <div className="backend-select__options">
          {BACKENDS.map((b) => (
            <button
              key={b}
              type="button"
              className={`backend-option ${backend === b ? "backend-option--active" : ""}`}
              onClick={() => setBackend(b)}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <button className="predict-button" onClick={handlePredict} disabled={!file || loading}>
        {loading ? "Running inference..." : "Predict"}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="result" key={result.total_latency_ms + result.predicted_label}>
          <p className="result__label">{result.predicted_label}</p>
          <p className="result__classid">class id {result.predicted_class_id}</p>

          <div className="result__grid">
            <div className="result__stat">
              <p className="result__stat-label">Backend</p>
              <p className="result__stat-value">{result.backend}</p>
            </div>
            <div className="result__stat">
              <p className="result__stat-label">Latency</p>
              <p className="result__stat-value">{result.total_latency_ms.toFixed(2)} ms</p>
            </div>
            <div className="result__stat">
              <p className="result__stat-label">Cache</p>
              <p className="result__stat-value">
                <span className={`cache-badge ${result.cache_hit ? "cache-badge--hit" : "cache-badge--miss"}`}>
                  {result.cache_hit ? "HIT" : "MISS"}
                </span>
              </p>
            </div>
            <div className="result__stat">
              <p className="result__stat-label">Batch size</p>
              <p className="result__stat-value">{result.batch_size ?? "—"}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
