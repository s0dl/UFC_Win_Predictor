import React, { useEffect, useMemo, useState } from "react";
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

function FighterSearchInput({ value, onChange, fighters }) {
  const [open, setOpen] = useState(false);
  const query = value.trim().toLowerCase();
  const matches = useMemo(() => {
    if (!fighters.length) return [];

    const ranked = fighters
      .filter((fighter) => fighter.name.toLowerCase().includes(query))
      .sort((left, right) => {
        const leftName = left.name.toLowerCase();
        const rightName = right.name.toLowerCase();
        const leftStarts = leftName.startsWith(query) ? 0 : 1;
        const rightStarts = rightName.startsWith(query) ? 0 : 1;
        return leftStarts - rightStarts || left.name.localeCompare(right.name);
      });

    return ranked.slice(0, 8);
  }, [fighters, query]);

  function selectFighter(fighter) {
    onChange(fighter);
    setOpen(false);
  }

  return (
    <div className="fighter-search">
      <input
        autoComplete="off"
        value={value}
        onBlur={() => setOpen(false)}
        onChange={(event) => {
          onChange(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        required
      />
      {open && matches.length > 0 && (
        <div className="fighter-menu">
          {matches.map((fighter) => (
            <button
              key={fighter.key}
              className="fighter-option"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => selectFighter(fighter.name)}
              type="button"
            >
              <span>{fighter.name}</span>
              <span>{fighter.prior_fights} fights</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function FighterCard({
  title,
  name,
  onNameChange,
  fighters,
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
        <FighterSearchInput fighters={fighters} value={name} onChange={onNameChange} />
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
  const [fighters, setFighters] = useState([]);
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

  useEffect(() => {
    let ignore = false;

    async function loadFighters() {
      try {
        const response = await fetch(`${apiUrl.replace(/\/$/, "")}/fighters`);
        const data = await response.json();
        if (!response.ok) return;
        if (!ignore) {
          const fighterOptions = Array.isArray(data.fighters)
            ? data.fighters
                .map((fighter) => {
                  if (typeof fighter === "string") {
                    return { name: fighter, key: fighter, prior_fights: 0 };
                  }
                  return {
                    name: fighter.name,
                    key: fighter.key || fighter.name,
                    prior_fights: fighter.prior_fights ?? 0,
                  };
                })
                .filter((fighter) => fighter.name)
            : [];
          setFighters(fighterOptions);
        }
      } catch {
        if (!ignore) setFighters([]);
      }
    }

    loadFighters();
    return () => {
      ignore = true;
    };
  }, [apiUrl]);

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
            fighters={fighters}
            openOdds={fighter1Open}
            onOpenOddsChange={setFighter1Open}
            closeOdds={fighter1Close}
            onCloseOddsChange={setFighter1Close}
          />
          <FighterCard
            title="Fighter 2"
            name={fighter2}
            onNameChange={setFighter2}
            fighters={fighters}
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

function App() {
  const [apiUrl] = useState(DEFAULT_API_URL);

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Fight night model</p>
          <h1>UFC Fight Predictor</h1>
        </div>
      </header>

      <PredictView apiUrl={apiUrl} />
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
