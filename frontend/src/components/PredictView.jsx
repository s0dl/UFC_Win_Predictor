import React from "react";
import { useEffect, useMemo, useState } from "react";
import FighterCard from "./FighterCard.jsx";
import ResultPanel from "./ResultPanel.jsx";
import { parseOdds } from "../utils/format.js";

function normalizeFighterOptions(fighters) {
  if (!Array.isArray(fighters)) return [];

  return fighters
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
    .filter((fighter) => fighter.name);
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
          setFighters(normalizeFighterOptions(data.fighters));
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

export default PredictView;
