import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function parseOdds(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.0%";
  return `${(value * 100).toFixed(1)}%`;
}

function ResultPanel({ result, error, loading }) {
  if (loading) {
    return (
      <section className="result-panel">
        <div className="status-line">Running model...</div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="result-panel">
        <div className="error-text">{error}</div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="result-panel empty">
        <div>Enter two fighters to get a prediction.</div>
      </section>
    );
  }

  return (
    <section className="result-panel">
      <div className="result-summary">
        <div>
          <span className="eyebrow">Predicted winner</span>
          <h2>{result.predicted_winner}</h2>
        </div>
        <strong>{formatPercent(result.predicted_winner_probability)}</strong>
      </div>

      <div className="probability-list">
        <ProbabilityBar
          name={result.fighter1}
          value={result.fighter1_win_probability}
          display={formatPercent(result.fighter1_win_probability)}
        />
        <ProbabilityBar
          name={result.fighter2}
          value={result.fighter2_win_probability}
          display={formatPercent(result.fighter2_win_probability)}
        />
      </div>
    </section>
  );
}

function ProbabilityBar({ name, display }) {
  return (
    <div className="probability-row">
      <div className="probability-label">
        <span>{name}</span>
        <span>{display}</span>
      </div>
      <div className="meter">
        <div className="meter-fill" style={{ width: display }} />
      </div>
    </div>
  );
}

function FighterCard({
  title,
  name,
  onNameChange,
  openOdds,
  onOpenOddsChange,
  closeOdds,
  onCloseOddsChange,
}) {
  return (
    <fieldset className="fighter-card">
      <legend>{title}</legend>
      <label>
        Name
        <input value={name} onChange={(event) => onNameChange(event.target.value)} required />
      </label>
      <div className="odds-grid">
        <label>
          Opening line
          <input
            inputMode="numeric"
            placeholder="-150"
            value={openOdds}
            onChange={(event) => onOpenOddsChange(event.target.value)}
          />
        </label>
        <label>
          Closing line
          <input
            inputMode="numeric"
            placeholder="130"
            value={closeOdds}
            onChange={(event) => onCloseOddsChange(event.target.value)}
          />
        </label>
      </div>
    </fieldset>
  );
}

function PredictView({ apiUrl }) {
  const [fighter1, setFighter1] = useState("Islam Makhachev");
  const [fighter2, setFighter2] = useState("Charles Oliveira");
  const [fighter1Open, setFighter1Open] = useState("");
  const [fighter2Open, setFighter2Open] = useState("");
  const [fighter1Close, setFighter1Close] = useState("");
  const [fighter2Close, setFighter2Close] = useState("");
  const [oddsInferred, setOddsInferred] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(
    () => fighter1.trim().length > 0 && fighter2.trim().length > 0 && !loading,
    [fighter1, fighter2, loading],
  );

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    const payload = {
      fighter1: fighter1.trim(),
      fighter2: fighter2.trim(),
      fighter1_open_odds: parseOdds(fighter1Open),
      fighter2_open_odds: parseOdds(fighter2Open),
      fighter1_close_odds: parseOdds(fighter1Close),
      fighter2_close_odds: parseOdds(fighter2Close),
      odds_inferred: oddsInferred ? 1 : 0,
    };

    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }
      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message || "Could not reach the prediction server.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="workspace">
      <form className="fight-form" onSubmit={handleSubmit}>
        <div className="fighter-grid">
          <FighterCard
            title="Fighter 1"
            name={fighter1}
            onNameChange={setFighter1}
            openOdds={fighter1Open}
            onOpenOddsChange={setFighter1Open}
            closeOdds={fighter1Close}
            onCloseOddsChange={setFighter1Close}
          />
          <FighterCard
            title="Fighter 2"
            name={fighter2}
            onNameChange={setFighter2}
            openOdds={fighter2Open}
            onOpenOddsChange={setFighter2Open}
            closeOdds={fighter2Close}
            onCloseOddsChange={setFighter2Close}
          />
        </div>

        <div className="actions">
          <label className="check-row">
            <input
              type="checkbox"
              checked={oddsInferred}
              onChange={(event) => setOddsInferred(event.target.checked)}
            />
            <span>Odds are inferred or unavailable</span>
          </label>
          <button disabled={!canSubmit} type="submit">
            {loading ? "Predicting..." : "Predict fight"}
          </button>
        </div>
      </form>

      <ResultPanel result={result} error={error} loading={loading} />
    </section>
  );
}

function RankingsView({ apiUrl }) {
  const [limit, setLimit] = useState(5000);
  const [rankings, setRankings] = useState([]);
  const [total, setTotal] = useState(0);
  const [benchmarkCount, setBenchmarkCount] = useState(10);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadRankings() {
    setLoading(true);
    setError("");
    try {
      const baseUrl = apiUrl.replace(/\/$/, "");
      const response = await fetch(`${baseUrl}/rankings?limit=${limit}&benchmark_count=${benchmarkCount}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not load rankings.");
      }
      setRankings(data.rankings || []);
      setTotal(data.total || 0);
    } catch (requestError) {
      setRankings([]);
      setTotal(0);
      setError(requestError.message || "Could not reach the prediction server.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rankings-view">
      <div className="ranking-toolbar">
        <div>
          <p className="eyebrow">Model leaderboard</p>
          <h2>All fighters ranked by benchmark score</h2>
        </div>
        <div className="ranking-controls">
          <label>
            Show
            <input
              inputMode="numeric"
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
            />
          </label>
          <label>
            Benchmarks
            <input
              inputMode="numeric"
              value={benchmarkCount}
              onChange={(event) => setBenchmarkCount(event.target.value)}
            />
          </label>
          <button onClick={loadRankings} disabled={loading}>
            {loading ? "Ranking..." : "Load rankings"}
          </button>
        </div>
      </div>

      <section className="rankings-panel">
        {error && <div className="error-text">{error}</div>}
        {!error && !loading && rankings.length === 0 && (
          <div className="status-line">Load rankings to see the model's fighter leaderboard.</div>
        )}
        {loading && <div className="status-line">Scoring fighters against benchmark panel...</div>}
        {!loading && rankings.length > 0 && (
          <>
            <div className="ranking-meta">
              Showing {rankings.length} of {total} rankable fighters.
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Fighter</th>
                    <th>Score</th>
                    <th>Benchmark wins</th>
                    <th>Prior fights</th>
                  </tr>
                </thead>
                <tbody>
                  {rankings.map((row) => (
                    <tr key={row.rank}>
                      <td>{row.rank}</td>
                      <td>{row.fighter}</td>
                      <td>{formatPercent(row.model_score)}</td>
                      <td>
                        {row.benchmark_wins}/{row.benchmark_count}
                      </td>
                      <td>{row.prior_fights}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </section>
  );
}

function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [activeView, setActiveView] = useState("predict");

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Fight model</p>
          <h1>UFC Fight Predictor</h1>
        </div>
        <label className="api-field">
          API URL
          <input value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} />
        </label>
      </header>

      <nav className="tabs" aria-label="Views">
        <button
          className={activeView === "predict" ? "active" : ""}
          onClick={() => setActiveView("predict")}
          type="button"
        >
          Predict
        </button>
        <button
          className={activeView === "rankings" ? "active" : ""}
          onClick={() => setActiveView("rankings")}
          type="button"
        >
          Rankings
        </button>
      </nav>

      {activeView === "predict" ? <PredictView apiUrl={apiUrl} /> : <RankingsView apiUrl={apiUrl} />}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
