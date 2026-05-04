"""
Build a no-leakage UFC fight modeling dataset from scraped UFCStats CSVs.

Each fight row uses only fighter information available before that event. Static
tale-of-the-tape fields are joined from fighter profiles, while performance
features are rolling aggregates built from prior fights.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["height_cm", "weight_in_kg", "reach_in_cm", "stance", "date_of_birth"]


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_text_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_date(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", format="mixed")


def parse_height_cm(value: object) -> float:
    if pd.isna(value) or str(value).strip() in {"", "--"}:
        return np.nan
    match = re.search(r"(\d+)'\s*(\d+)", str(value))
    if not match:
        return np.nan
    return round(((int(match.group(1)) * 12) + int(match.group(2))) * 2.54, 2)


def parse_weight_kg(value: object) -> float:
    if pd.isna(value) or str(value).strip() in {"", "--"}:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return round(float(match.group(1)) * 0.45359237, 2) if match else np.nan


def parse_reach_cm(value: object) -> float:
    if pd.isna(value) or str(value).strip() in {"", "--"}:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return round(float(match.group(1)) * 2.54, 2) if match else np.nan


def parse_of_value(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return 0.0, 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)", str(value))
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def parse_ctrl_seconds(value: object) -> float:
    if pd.isna(value) or str(value).strip() in {"", "--", "---"}:
        return 0.0
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return 0.0
    return (int(parts[0]) * 60) + int(parts[1])


def parse_round_lengths(time_format: object) -> list[float]:
    if pd.isna(time_format):
        return []
    match = re.search(r"\(([^)]+)\)", str(time_format))
    if not match:
        return []
    return [float(part) for part in re.findall(r"\d+(?:\.\d+)?", match.group(1))]


def parse_elapsed_minutes(value: object) -> float:
    if pd.isna(value):
        return 0.0
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return 0.0
    return int(parts[0]) + (int(parts[1]) / 60.0)


def fight_duration_minutes(row: pd.Series) -> float:
    round_number = int(row["ROUND"]) if not pd.isna(row["ROUND"]) else 1
    elapsed = parse_elapsed_minutes(row["TIME"])
    round_lengths = parse_round_lengths(row["TIME FORMAT"])
    if round_lengths:
        previous = sum(round_lengths[: max(round_number - 1, 0)])
    else:
        previous = 5.0 * max(round_number - 1, 0)
    return previous + elapsed


def safe_rate(numerator: float, denominator: float, multiplier: float = 1.0) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * multiplier


def american_to_implied_prob(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


@dataclass
class FighterState:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    no_contests: int = 0
    ko_tko_wins: int = 0
    submission_wins: int = 0
    decision_wins: int = 0
    ko_tko_losses: int = 0
    submission_losses: int = 0
    decision_losses: int = 0
    fight_time_min: float = 0.0
    sig_landed: float = 0.0
    sig_attempted: float = 0.0
    sig_absorbed: float = 0.0
    sig_abs_attempted: float = 0.0
    total_landed: float = 0.0
    total_attempted: float = 0.0
    td_landed: float = 0.0
    td_attempted: float = 0.0
    td_absorbed: float = 0.0
    td_abs_attempted: float = 0.0
    sub_attempts: float = 0.0
    kd_for: float = 0.0
    kd_against: float = 0.0
    reversals: float = 0.0
    ctrl_seconds: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    days_since_last_fight: float = np.nan
    last_fight_date: pd.Timestamp | None = None
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=5))

    def features(self, prefix: str, event_date: pd.Timestamp, static: dict[str, object]) -> dict[str, object]:
        decided = self.wins + self.losses + self.draws
        fights = decided + self.no_contests
        recent_3 = list(self.recent_results)[-3:]
        recent_5 = list(self.recent_results)[-5:]
        days_since = np.nan
        if self.last_fight_date is not None and not pd.isna(event_date):
            days_since = (event_date - self.last_fight_date).days

        dob = static.get("date_of_birth", pd.NaT)
        age = np.nan
        if not pd.isna(dob) and not pd.isna(event_date):
            age = (event_date - dob).days / 365.25

        return {
            f"{prefix}_height_cm": static.get("height_cm", np.nan),
            f"{prefix}_weight_in_kg": static.get("weight_in_kg", np.nan),
            f"{prefix}_reach_in_cm": static.get("reach_in_cm", np.nan),
            f"{prefix}_age": age,
            f"{prefix}_stance": static.get("stance", "Unknown"),
            f"{prefix}_fights_before": fights,
            f"{prefix}_wins_before": self.wins,
            f"{prefix}_losses_before": self.losses,
            f"{prefix}_draws_before": self.draws,
            f"{prefix}_no_contests_before": self.no_contests,
            f"{prefix}_win_rate_before": safe_rate(self.wins, decided),
            f"{prefix}_ko_tko_win_rate_before": safe_rate(self.ko_tko_wins, self.wins),
            f"{prefix}_submission_win_rate_before": safe_rate(self.submission_wins, self.wins),
            f"{prefix}_decision_win_rate_before": safe_rate(self.decision_wins, self.wins),
            f"{prefix}_ko_tko_loss_rate_before": safe_rate(self.ko_tko_losses, self.losses),
            f"{prefix}_submission_loss_rate_before": safe_rate(self.submission_losses, self.losses),
            f"{prefix}_decision_loss_rate_before": safe_rate(self.decision_losses, self.losses),
            f"{prefix}_slpm_before": safe_rate(self.sig_landed, self.fight_time_min),
            f"{prefix}_striking_accuracy_before": safe_rate(self.sig_landed, self.sig_attempted, 100.0),
            f"{prefix}_sapm_before": safe_rate(self.sig_absorbed, self.fight_time_min),
            f"{prefix}_striking_defense_before": 100.0 - safe_rate(self.sig_absorbed, self.sig_abs_attempted, 100.0),
            f"{prefix}_td_avg_before": safe_rate(self.td_landed, self.fight_time_min, 15.0),
            f"{prefix}_td_accuracy_before": safe_rate(self.td_landed, self.td_attempted, 100.0),
            f"{prefix}_td_defense_before": 100.0 - safe_rate(self.td_absorbed, self.td_abs_attempted, 100.0),
            f"{prefix}_sub_avg_before": safe_rate(self.sub_attempts, self.fight_time_min, 15.0),
            f"{prefix}_kd_avg_before": safe_rate(self.kd_for, self.fight_time_min, 15.0),
            f"{prefix}_kd_absorbed_avg_before": safe_rate(self.kd_against, self.fight_time_min, 15.0),
            f"{prefix}_ctrl_avg_before": safe_rate(self.ctrl_seconds / 60.0, self.fight_time_min, 15.0),
            f"{prefix}_current_win_streak_before": self.current_win_streak,
            f"{prefix}_current_loss_streak_before": self.current_loss_streak,
            f"{prefix}_recent_3_win_rate_before": safe_rate(sum(recent_3), len(recent_3)),
            f"{prefix}_recent_5_win_rate_before": safe_rate(sum(recent_5), len(recent_5)),
            f"{prefix}_days_since_last_fight": days_since,
        }

    def update_result(self, result: str, method: str, event_date: pd.Timestamp) -> None:
        method = str(method)
        if result == "W":
            self.wins += 1
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.recent_results.append(1)
            if "KO/TKO" in method:
                self.ko_tko_wins += 1
            elif "SUB" in method:
                self.submission_wins += 1
            elif "Decision" in method:
                self.decision_wins += 1
        elif result == "L":
            self.losses += 1
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.recent_results.append(0)
            if "KO/TKO" in method:
                self.ko_tko_losses += 1
            elif "SUB" in method:
                self.submission_losses += 1
            elif "Decision" in method:
                self.decision_losses += 1
        elif result == "D":
            self.draws += 1
            self.current_win_streak = 0
            self.current_loss_streak = 0
        else:
            self.no_contests += 1

        if not pd.isna(event_date):
            self.last_fight_date = event_date

    def update_stats(self, own: dict[str, float], opponent: dict[str, float], duration_min: float) -> None:
        self.fight_time_min += duration_min
        self.sig_landed += own["sig_landed"]
        self.sig_attempted += own["sig_attempted"]
        self.sig_absorbed += opponent["sig_landed"]
        self.sig_abs_attempted += opponent["sig_attempted"]
        self.total_landed += own["total_landed"]
        self.total_attempted += own["total_attempted"]
        self.td_landed += own["td_landed"]
        self.td_attempted += own["td_attempted"]
        self.td_absorbed += opponent["td_landed"]
        self.td_abs_attempted += opponent["td_attempted"]
        self.sub_attempts += own["sub_attempts"]
        self.kd_for += own["kd"]
        self.kd_against += opponent["kd"]
        self.reversals += own["reversals"]
        self.ctrl_seconds += own["ctrl_seconds"]


def load_static_fighters(scrape_dir: Path) -> dict[str, dict[str, object]]:
    tott = pd.read_csv(scrape_dir / "ufc_fighter_tott.csv")
    details = pd.read_csv(scrape_dir / "ufc_fighter_details.csv")
    tott["name_key"] = tott["FIGHTER"].map(normalize_name)
    details["name_key"] = (details["FIRST"].fillna("") + " " + details["LAST"].fillna("")).map(normalize_name)
    merged = tott.merge(details[["name_key", "NICKNAME"]], on="name_key", how="left")
    merged["height_cm"] = merged["HEIGHT"].map(parse_height_cm)
    merged["weight_in_kg"] = merged["WEIGHT"].map(parse_weight_kg)
    merged["reach_in_cm"] = merged["REACH"].map(parse_reach_cm)
    merged["stance"] = merged["STANCE"].fillna("Unknown").replace({"": "Unknown", "--": "Unknown"})
    merged["date_of_birth"] = pd.to_datetime(merged["DOB"].replace("--", np.nan), errors="coerce")
    merged["nickname"] = merged["NICKNAME"].fillna("")
    merged = merged.sort_values(["name_key", "date_of_birth"], na_position="last")
    merged = merged.drop_duplicates("name_key", keep="first")
    return merged.set_index("name_key")[STATIC_COLUMNS + ["nickname"]].to_dict("index")


def aggregate_fight_stats(stats: pd.DataFrame) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    rows = {}
    for (event, bout, fighter), group in stats.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False):
        sig = group["SIG.STR."].map(parse_of_value)
        total = group["TOTAL STR."].map(parse_of_value)
        td = group["TD"].map(parse_of_value)
        rows.setdefault((event, bout), {})[normalize_name(fighter)] = {
            "sig_landed": sum(x[0] for x in sig),
            "sig_attempted": sum(x[1] for x in sig),
            "total_landed": sum(x[0] for x in total),
            "total_attempted": sum(x[1] for x in total),
            "td_landed": sum(x[0] for x in td),
            "td_attempted": sum(x[1] for x in td),
            "sub_attempts": float(group["SUB.ATT"].fillna(0).sum()),
            "kd": float(group["KD"].fillna(0).sum()),
            "reversals": float(group["REV."].fillna(0).sum()),
            "ctrl_seconds": float(group["CTRL"].map(parse_ctrl_seconds).sum()),
        }
    return rows


def add_diff_features(row: dict[str, object], feature_names: Iterable[str]) -> None:
    for feature in feature_names:
        left = row.get(f"f1_{feature}")
        right = row.get(f"f2_{feature}")
        if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(right, (int, float, np.integer, np.floating)):
            row[f"diff_{feature}"] = left - right


def normalize_odds_file(odds_path: Path) -> pd.DataFrame:
    odds = pd.read_csv(odds_path)
    lower_map = {col: col.lower().strip() for col in odds.columns}
    odds = odds.rename(columns=lower_map)

    rename_candidates = {
        "event": ["event", "event_name"],
        "date": ["date", "event_date"],
        "fighter1": ["fighter1", "fighter_1", "r_fighter", "red_fighter"],
        "fighter2": ["fighter2", "fighter_2", "b_fighter", "blue_fighter"],
        "f1_odds": ["f1_odds", "fighter1_odds", "r_odds", "red_odds"],
        "f2_odds": ["f2_odds", "fighter2_odds", "b_odds", "blue_odds"],
    }

    chosen = {}
    for canonical, candidates in rename_candidates.items():
        for candidate in candidates:
            if candidate in odds.columns:
                chosen[candidate] = canonical
                break
    odds = odds.rename(columns=chosen)

    required = {"date", "fighter1", "fighter2", "f1_odds", "f2_odds"}
    missing = sorted(required - set(odds.columns))
    if missing:
        raise ValueError(f"Odds CSV is missing required columns after normalization: {missing}")

    odds["event_date"] = pd.to_datetime(odds["date"], errors="coerce")
    odds["fighter1_key"] = odds["fighter1"].map(normalize_name)
    odds["fighter2_key"] = odds["fighter2"].map(normalize_name)
    odds["f1_odds"] = pd.to_numeric(odds["f1_odds"], errors="coerce")
    odds["f2_odds"] = pd.to_numeric(odds["f2_odds"], errors="coerce")
    return odds


def build_odds_lookup(odds_path: Path | None) -> dict[tuple[pd.Timestamp, frozenset[str]], dict[str, float]]:
    if odds_path is None:
        return {}
    odds = normalize_odds_file(odds_path)
    lookup = {}
    for _, row in odds.iterrows():
        if pd.isna(row["event_date"]):
            continue
        key = (row["event_date"].normalize(), frozenset([row["fighter1_key"], row["fighter2_key"]]))
        lookup[key] = {
            row["fighter1_key"]: row["f1_odds"],
            row["fighter2_key"]: row["f2_odds"],
        }
    return lookup


def attach_odds_features(row: dict[str, object], odds_lookup: dict[tuple[pd.Timestamp, frozenset[str]], dict[str, float]]) -> None:
    if not odds_lookup:
        return
    key = (row["event_date"].normalize(), frozenset([row["fighter1_key"], row["fighter2_key"]]))
    odds = odds_lookup.get(key)
    if not odds:
        row.update(
            {
                "f1_moneyline": np.nan,
                "f2_moneyline": np.nan,
                "f1_implied_prob": np.nan,
                "f2_implied_prob": np.nan,
                "diff_implied_prob": np.nan,
                "f1_is_favorite": np.nan,
            }
        )
        return
    f1_odds = odds.get(row["fighter1_key"], np.nan)
    f2_odds = odds.get(row["fighter2_key"], np.nan)
    f1_prob = american_to_implied_prob(f1_odds)
    f2_prob = american_to_implied_prob(f2_odds)
    row["f1_moneyline"] = f1_odds
    row["f2_moneyline"] = f2_odds
    row["f1_implied_prob"] = f1_prob
    row["f2_implied_prob"] = f2_prob
    row["diff_implied_prob"] = f1_prob - f2_prob if not pd.isna(f1_prob) and not pd.isna(f2_prob) else np.nan
    row["f1_is_favorite"] = float(f1_odds < f2_odds) if not pd.isna(f1_odds) and not pd.isna(f2_odds) else np.nan


def build_dataset(scrape_dir: Path, odds_csv: Path | None = None) -> pd.DataFrame:
    events = pd.read_csv(scrape_dir / "ufc_event_details.csv")
    results = pd.read_csv(scrape_dir / "ufc_fight_results.csv")
    stats = pd.read_csv(scrape_dir / "ufc_fight_stats.csv")

    events["event_key"] = events["EVENT"].map(normalize_text_key)
    results["event_key"] = results["EVENT"].map(normalize_text_key)
    stats["EVENT"] = stats["EVENT"].map(normalize_text_key)
    stats["BOUT"] = stats["BOUT"].map(normalize_text_key)
    results["EVENT"] = results["EVENT"].map(normalize_text_key)
    events["event_date"] = events["DATE"].map(parse_date)
    results = results.merge(events[["event_key", "event_date", "LOCATION"]], on="event_key", how="left")
    results["original_order"] = np.arange(len(results))
    results["duration_min"] = results.apply(fight_duration_minutes, axis=1)

    static = load_static_fighters(scrape_dir)
    fight_stats = aggregate_fight_stats(stats)
    odds_lookup = build_odds_lookup(odds_csv)
    states: defaultdict[str, FighterState] = defaultdict(FighterState)
    rows: list[dict[str, object]] = []

    sorted_results = results.sort_values(["event_date", "EVENT", "original_order"], ascending=[True, True, False])
    feature_names = [
        "height_cm",
        "weight_in_kg",
        "reach_in_cm",
        "age",
        "fights_before",
        "wins_before",
        "losses_before",
        "draws_before",
        "no_contests_before",
        "win_rate_before",
        "ko_tko_win_rate_before",
        "submission_win_rate_before",
        "decision_win_rate_before",
        "ko_tko_loss_rate_before",
        "submission_loss_rate_before",
        "decision_loss_rate_before",
        "slpm_before",
        "striking_accuracy_before",
        "sapm_before",
        "striking_defense_before",
        "td_avg_before",
        "td_accuracy_before",
        "td_defense_before",
        "sub_avg_before",
        "kd_avg_before",
        "kd_absorbed_avg_before",
        "ctrl_avg_before",
        "current_win_streak_before",
        "current_loss_streak_before",
        "recent_3_win_rate_before",
        "recent_5_win_rate_before",
        "days_since_last_fight",
    ]

    for (_, event), event_results in sorted_results.groupby(["event_date", "EVENT"], sort=False, dropna=False):
        event_updates = []
        for _, fight in event_results.iterrows():
            fighters = [normalize_name(name) for name in str(fight["BOUT"]).split(" vs. ")]
            outcomes = str(fight["OUTCOME"]).split("/")
            if len(fighters) != 2 or len(outcomes) != 2:
                continue

            fighter1, fighter2 = fighters
            outcome = 1 if outcomes[0] == "W" else 0 if outcomes[1] == "W" else np.nan
            row = {
                "event": fight["EVENT"],
                "event_date": fight["event_date"],
                "location": fight["LOCATION"],
                "bout": fight["BOUT"],
                "fighter1": fighter1,
                "fighter2": fighter2,
                "fighter1_key": fighter1,
                "fighter2_key": fighter2,
                "weightclass": fight["WEIGHTCLASS"],
                "method": fight["METHOD"],
                "scheduled_time_format": fight["TIME FORMAT"],
                "outcome": outcome,
            }
            row.update(states[fighter1].features("f1", fight["event_date"], static.get(fighter1, {})))
            row.update(states[fighter2].features("f2", fight["event_date"], static.get(fighter2, {})))
            add_diff_features(row, feature_names)
            attach_odds_features(row, odds_lookup)
            rows.append(row)
            event_updates.append((fight, fighters, outcomes))

        for fight, fighters, outcomes in event_updates:
            fight_key = (fight["EVENT"], fight["BOUT"])
            stats_for_fight = fight_stats.get(fight_key, {})
            zero_stats = {
                "sig_landed": 0.0,
                "sig_attempted": 0.0,
                "total_landed": 0.0,
                "total_attempted": 0.0,
                "td_landed": 0.0,
                "td_attempted": 0.0,
                "sub_attempts": 0.0,
                "kd": 0.0,
                "reversals": 0.0,
                "ctrl_seconds": 0.0,
            }
            f1, f2 = fighters
            f1_stats = stats_for_fight.get(f1, zero_stats)
            f2_stats = stats_for_fight.get(f2, zero_stats)
            states[f1].update_result(outcomes[0], fight["METHOD"], fight["event_date"])
            states[f2].update_result(outcomes[1], fight["METHOD"], fight["event_date"])
            states[f1].update_stats(f1_stats, f2_stats, fight["duration_min"])
            states[f2].update_stats(f2_stats, f1_stats, fight["duration_min"])

    df = pd.DataFrame(rows)
    if "outcome" in df.columns:
        df = df[df["outcome"].isin([0, 1])].copy()
        df["outcome"] = df["outcome"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape-dir", default="scrape_ufc_stats", type=Path)
    parser.add_argument("--output", default="fight_data_historical_no_leakage.csv", type=Path)
    parser.add_argument("--odds-csv", default=None, type=Path)
    args = parser.parse_args()

    df = build_dataset(args.scrape_dir, args.odds_csv)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {df.shape}")
    print(f"Outcome distribution: {df['outcome'].value_counts().to_dict()}")
    if "f1_moneyline" in df.columns:
        print(f"Rows with odds: {df['f1_moneyline'].notna().sum()} / {len(df)}")


if __name__ == "__main__":
    main()
