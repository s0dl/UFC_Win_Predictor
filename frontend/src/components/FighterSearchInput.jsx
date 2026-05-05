import React from "react";
import { useMemo, useState } from "react";

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

export default FighterSearchInput;
