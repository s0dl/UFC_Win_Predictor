export function parseOdds(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.0%";
  return `${(value * 100).toFixed(1)}%`;
}
