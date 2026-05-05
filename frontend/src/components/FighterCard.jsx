import React from "react";
import FighterSearchInput from "./FighterSearchInput.jsx";

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
            placeholder="+130"
            value={closeOdds}
            onChange={(event) => onCloseOddsChange(event.target.value)}
          />
        </label>
      </div>
    </fieldset>
  );
}

export default FighterCard;
