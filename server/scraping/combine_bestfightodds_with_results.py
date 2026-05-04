"""
Join BestFightOdds open/close moneylines onto UFCStats fight results.

The output keeps the original UFCStats bout order. Odds columns are aligned to
the first and second fighter listed in BOUT, regardless of BestFightOdds order.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from scraping_common import CACHE_DIR, DATA_DIR


def clean_name(value: object) -> str:
    text = str(value).lower().strip().replace(".", "").replace("'", "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sorted_pair_key(left: str, right: str) -> str:
    return "|".join(sorted([left, right]))


def american_to_implied_probability(odds: float) -> float | None:
    if pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return None


def probability_to_american(probability: float) -> int:
    probability = min(max(float(probability), 0.01), 0.99)
    if probability >= 0.5:
        return int(round(-(probability / (1.0 - probability)) * 100.0))
    return int(round(((1.0 - probability) / probability) * 100.0))


def no_vig_pair_probability(f1_odds: float, f2_odds: float) -> tuple[float, float] | tuple[None, None]:
    p1 = american_to_implied_probability(f1_odds)
    p2 = american_to_implied_probability(f2_odds)
    if p1 is None or p2 is None or (p1 + p2) <= 0:
        return None, None
    return p1 / (p1 + p2), p2 / (p1 + p2)


def fighter_market_strengths(odds: pd.DataFrame, open_or_close: str) -> dict[str, float]:
    f1_col = f"f1_{open_or_close}"
    f2_col = f"f2_{open_or_close}"
    strengths: dict[str, list[float]] = {}
    for row in odds.itertuples(index=False):
        p1, p2 = no_vig_pair_probability(getattr(row, f1_col), getattr(row, f2_col))
        if p1 is None or p2 is None:
            continue
        strengths.setdefault(row.odds_f1_clean, []).append(p1)
        strengths.setdefault(row.odds_f2_clean, []).append(p2)
    return {fighter: sum(values) / len(values) for fighter, values in strengths.items()}


def infer_missing_odds(merged: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    open_strength = fighter_market_strengths(odds, "open")
    close_strength = fighter_market_strengths(odds, "close")

    missing = ~merged["odds_matched"]
    merged["odds_inferred"] = missing

    for index, row in merged.loc[missing].iterrows():
        f1 = row["result_f1_clean"]
        f2 = row["result_f2_clean"]

        f1_open_strength = open_strength.get(f1, 0.5)
        f2_open_strength = open_strength.get(f2, 0.5)
        f1_close_strength = close_strength.get(f1, 0.5)
        f2_close_strength = close_strength.get(f2, 0.5)

        open_total = f1_open_strength + f2_open_strength
        close_total = f1_close_strength + f2_close_strength
        f1_open_probability = f1_open_strength / open_total if open_total else 0.5
        f1_close_probability = f1_close_strength / close_total if close_total else 0.5

        merged.at[index, "f1_open_odds"] = probability_to_american(f1_open_probability)
        merged.at[index, "f2_open_odds"] = probability_to_american(1.0 - f1_open_probability)
        merged.at[index, "f1_close_odds"] = probability_to_american(f1_close_probability)
        merged.at[index, "f2_close_odds"] = probability_to_american(1.0 - f1_close_probability)
        merged.at[index, "odds_event"] = "inferred_from_fighter_market_history"

    return merged


def build_results_with_odds(results_path: Path, events_path: Path, odds_path: Path, output_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    events = pd.read_csv(events_path)
    odds = pd.read_csv(odds_path)

    results["EVENT"] = results["EVENT"].str.strip()
    results["BOUT"] = results["BOUT"].str.strip()
    events["EVENT"] = events["EVENT"].str.strip()

    results = results.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    results["fight_date"] = pd.to_datetime(results["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    results = results.drop(columns=["DATE"])

    result_fighters = results["BOUT"].str.split(" vs. ", n=1, expand=True)
    results["result_f1_clean"] = result_fighters[0].map(clean_name)
    results["result_f2_clean"] = result_fighters[1].map(clean_name)
    results["pair_key"] = results.apply(lambda row: sorted_pair_key(row["result_f1_clean"], row["result_f2_clean"]), axis=1)

    odds["fight_date"] = pd.to_datetime(odds["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds["odds_f1_clean"] = odds["fighter1"].map(clean_name)
    odds["odds_f2_clean"] = odds["fighter2"].map(clean_name)
    odds["pair_key"] = odds.apply(lambda row: sorted_pair_key(row["odds_f1_clean"], row["odds_f2_clean"]), axis=1)
    for col in ["f1_open", "f2_open", "f1_close", "f2_close"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")

    odds_cols = [
        "fight_date",
        "pair_key",
        "event",
        "fighter1",
        "fighter2",
        "odds_f1_clean",
        "odds_f2_clean",
        "f1_open",
        "f2_open",
        "f1_close",
        "f2_close",
        "f1_open_ts",
        "f2_open_ts",
        "f1_close_ts",
        "f2_close_ts",
        "f1_history_points",
        "f2_history_points",
        "matchup_id",
        "source_url",
    ]
    merged = results.merge(odds[odds_cols], on=["fight_date", "pair_key"], how="left")

    result_f1_is_bfo_f1 = merged["result_f1_clean"] == merged["odds_f1_clean"]
    result_f1_is_bfo_f2 = merged["result_f1_clean"] == merged["odds_f2_clean"]
    matched = result_f1_is_bfo_f1 | result_f1_is_bfo_f2

    merged["f1_open_odds"] = pd.NA
    merged["f2_open_odds"] = pd.NA
    merged["f1_close_odds"] = pd.NA
    merged["f2_close_odds"] = pd.NA
    merged["f1_open_odds_ts"] = pd.NA
    merged["f2_open_odds_ts"] = pd.NA
    merged["f1_close_odds_ts"] = pd.NA
    merged["f2_close_odds_ts"] = pd.NA
    merged["f1_odds_history_points"] = pd.NA
    merged["f2_odds_history_points"] = pd.NA

    merged.loc[result_f1_is_bfo_f1, "f1_open_odds"] = merged.loc[result_f1_is_bfo_f1, "f1_open"]
    merged.loc[result_f1_is_bfo_f1, "f2_open_odds"] = merged.loc[result_f1_is_bfo_f1, "f2_open"]
    merged.loc[result_f1_is_bfo_f1, "f1_close_odds"] = merged.loc[result_f1_is_bfo_f1, "f1_close"]
    merged.loc[result_f1_is_bfo_f1, "f2_close_odds"] = merged.loc[result_f1_is_bfo_f1, "f2_close"]
    merged.loc[result_f1_is_bfo_f1, "f1_open_odds_ts"] = merged.loc[result_f1_is_bfo_f1, "f1_open_ts"]
    merged.loc[result_f1_is_bfo_f1, "f2_open_odds_ts"] = merged.loc[result_f1_is_bfo_f1, "f2_open_ts"]
    merged.loc[result_f1_is_bfo_f1, "f1_close_odds_ts"] = merged.loc[result_f1_is_bfo_f1, "f1_close_ts"]
    merged.loc[result_f1_is_bfo_f1, "f2_close_odds_ts"] = merged.loc[result_f1_is_bfo_f1, "f2_close_ts"]
    merged.loc[result_f1_is_bfo_f1, "f1_odds_history_points"] = merged.loc[result_f1_is_bfo_f1, "f1_history_points"]
    merged.loc[result_f1_is_bfo_f1, "f2_odds_history_points"] = merged.loc[result_f1_is_bfo_f1, "f2_history_points"]

    merged.loc[result_f1_is_bfo_f2, "f1_open_odds"] = merged.loc[result_f1_is_bfo_f2, "f2_open"]
    merged.loc[result_f1_is_bfo_f2, "f2_open_odds"] = merged.loc[result_f1_is_bfo_f2, "f1_open"]
    merged.loc[result_f1_is_bfo_f2, "f1_close_odds"] = merged.loc[result_f1_is_bfo_f2, "f2_close"]
    merged.loc[result_f1_is_bfo_f2, "f2_close_odds"] = merged.loc[result_f1_is_bfo_f2, "f1_close"]
    merged.loc[result_f1_is_bfo_f2, "f1_open_odds_ts"] = merged.loc[result_f1_is_bfo_f2, "f2_open_ts"]
    merged.loc[result_f1_is_bfo_f2, "f2_open_odds_ts"] = merged.loc[result_f1_is_bfo_f2, "f1_open_ts"]
    merged.loc[result_f1_is_bfo_f2, "f1_close_odds_ts"] = merged.loc[result_f1_is_bfo_f2, "f2_close_ts"]
    merged.loc[result_f1_is_bfo_f2, "f2_close_odds_ts"] = merged.loc[result_f1_is_bfo_f2, "f1_close_ts"]
    merged.loc[result_f1_is_bfo_f2, "f1_odds_history_points"] = merged.loc[result_f1_is_bfo_f2, "f2_history_points"]
    merged.loc[result_f1_is_bfo_f2, "f2_odds_history_points"] = merged.loc[result_f1_is_bfo_f2, "f1_history_points"]

    merged["odds_matched"] = matched
    merged["odds_event"] = merged["event"]
    merged["odds_matchup_id"] = merged["matchup_id"]
    merged["odds_source_url"] = merged["source_url"]
    merged = infer_missing_odds(merged, odds)

    helper_cols = [
        "result_f1_clean",
        "result_f2_clean",
        "pair_key",
        "event",
        "fighter1",
        "fighter2",
        "odds_f1_clean",
        "odds_f2_clean",
        "f1_open",
        "f2_open",
        "f1_close",
        "f2_close",
        "f1_open_ts",
        "f2_open_ts",
        "f1_close_ts",
        "f2_close_ts",
        "f1_history_points",
        "f2_history_points",
        "matchup_id",
        "source_url",
    ]
    merged = merged.drop(columns=[col for col in helper_cols if col in merged.columns])
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=CACHE_DIR / "ufc_fight_results.csv")
    parser.add_argument("--events", type=Path, default=CACHE_DIR / "ufc_event_details.csv")
    parser.add_argument("--odds", type=Path, default=CACHE_DIR / "bestfightodds_moneylines_full_open_close.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "ufc_fight_results_with_odds.csv")
    args = parser.parse_args()

    merged = build_results_with_odds(args.results, args.events, args.odds, args.output)
    matched = int(merged["odds_matched"].sum())
    inferred = int(merged["odds_inferred"].sum())
    print(f"Wrote {args.output} with shape {merged.shape}")
    print(f"Matched odds for {matched}/{len(merged)} fights ({matched / len(merged):.1%})")
    print(f"Inferred odds for {inferred}/{len(merged)} fights ({inferred / len(merged):.1%})")


if __name__ == "__main__":
    main()
