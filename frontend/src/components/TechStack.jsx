// Purely a visual "here's the stack" strip for the demo video - not
// wired to any live data, unlike the rest of this app's panels.
const STACK = [
  { name: "React", color: "#61dafb" },
  { name: "FastAPI", color: "#22d3ee" },
  { name: "PyTorch", color: "#ee4c2c" },
  { name: "ONNX Runtime", color: "#8b93a5" },
  { name: "TorchScript", color: "#ee4c2c" },
  { name: "Redis", color: "#dc382d" },
  { name: "PostgreSQL", color: "#4f8cff" },
  { name: "Prometheus", color: "#e6522c" },
  { name: "Grafana", color: "#f5a623" },
];

export default function TechStack() {
  return (
    <div className="techstack">
      {STACK.map((item) => (
        <span className="techstack__pill" key={item.name}>
          <span className="techstack__pill-dot" style={{ background: item.color }} />
          {item.name}
        </span>
      ))}
    </div>
  );
}
