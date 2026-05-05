import React from "react";

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

export default ProbabilityBar;
