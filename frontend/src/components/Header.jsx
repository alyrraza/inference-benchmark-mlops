export default function Header({ healthy }) {
  return (
    <header className="header">
      <div className="header__brand">
        <div className="header__mark">IB</div>
        <div>
          <h1 className="header__title">InferBench</h1>
          <p className="header__subtitle">
            Model inference optimization &amp; observability platform
          </p>
        </div>
      </div>
      <div className="header__badge">
        <span
          className="header__badge-dot"
          style={{ background: healthy ? "var(--success)" : "var(--error)" }}
        />
        {healthy ? "API connected" : "API unreachable"}
      </div>
    </header>
  );
}
