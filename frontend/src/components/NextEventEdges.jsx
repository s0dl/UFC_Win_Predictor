import React from "react";
import { useEffect, useMemo, useState } from "react";
import { formatOdds, formatPercent, formatSignedPercent } from "../utils/format.js";

function bestEdgeClass(edge) {
  if (typeof edge !== "number") return "";
  if (edge >= 0.05) return "edge-positive";
  if (edge < 0) return "edge-negative";
  return "";
}

function NextEventEdges({ apiUrl }) {
  const [eventData, setEventData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [collapsed, setCollapsed] = useState(false);

  async function loadEdges() {
    setLoading(true);
    setError("");

    try {
      const baseUrl = apiUrl.replace(/\/$/, "");
      const response = await fetch(`${baseUrl}/next-event/edges`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load the next UFC event.");
      setEventData(data);
    } catch (requestError) {
      setError(requestError.message || "Could not load the next UFC event.");
      setEventData(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadEdges(false);
  }, [apiUrl]);

  const fights = useMemo(() => eventData?.fights || [], [eventData]);

  return (
    <section className="event-panel">
      <div className="event-header">
        <div>
          <p className="eyebrow">Next UFC event edges</p>
          <h2>{eventData?.event || "Loading upcoming card"}</h2>
          {eventData?.date ? <p className="event-date">{eventData.date}</p> : null}
        </div>
        <button type="button" onClick={() => setCollapsed((current) => !current)}>
          {collapsed ? "Show card" : "Hide card"}
        </button>
      </div>

      {!collapsed && error ? <div className="error-text event-error">{error}</div> : null}
      {!collapsed && !error && loading && !eventData ? (
        <div className="status-line event-status">Loading current lines...</div>
      ) : null}

      {!collapsed && !error && fights.length > 0 ? (
        <div className="edge-table" role="table" aria-label="Next UFC event model edges">
          <div className="edge-row edge-head" role="row">
            <span>Fight</span>
            <span>Open</span>
            <span>Current</span>
            <span>Model</span>
            <span>Edge</span>
          </div>
          {[...fights].sort((a, b) => (a.importance_order ?? 999) - (b.importance_order ?? 999)).map((fight) => {
            const edge = fight.recommended_edge?.edge;
            const predicted = fight.prediction?.predicted_winner;
            return (
              <div className="edge-row" role="row" key={`${fight.importance_order}-${fight.fighter1}-${fight.fighter2}`}>
                <div className="fight-cell">
                  <strong>{fight.fighter1}</strong>
                  <span>vs {fight.fighter2}</span>
                </div>
                <div>
                  <span>{formatOdds(fight.fighter1_open_odds)}</span>
                  <span>{formatOdds(fight.fighter2_open_odds)}</span>
                </div>
                <div>
                  <span>{formatOdds(fight.fighter1_current_odds)}</span>
                  <span>{formatOdds(fight.fighter2_current_odds)}</span>
                </div>
                <div>
                  {fight.prediction ? (
                    <>
                      <strong>{predicted}</strong>
                      <span>{formatPercent(fight.prediction.predicted_winner_probability)}</span>
                    </>
                  ) : (
                    <span className="muted">{fight.model_error || "No model match"}</span>
                  )}
                </div>
                <div className={bestEdgeClass(edge)}>
                  {fight.recommended_edge ? (
                    <>
                      <strong>{fight.recommended_side}</strong>
                      <span>{formatSignedPercent(edge)}</span>
                      <small>Kelly {formatPercent(fight.recommended_edge.kelly_fraction)}</small>
                    </>
                  ) : (
                    <span className="muted">No line</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

export default NextEventEdges;
