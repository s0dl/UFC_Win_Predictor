import React from "react";
import ProbabilityBar from "./ProbabilityBar.jsx";
import { formatPercent } from "../utils/format.js";

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

export default ResultPanel;
