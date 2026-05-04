"""Update the legacy PyTorch-style server/models/data/ufc_fight_data.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scraping_common import DATA_DIR, write_csv_atomic
from update_ufc_fight_stats import DEFAULT_RESULTS_OUTPUT, update_ufcstats_fight_data


DEFAULT_FIGHTER_STATS = DATA_DIR / "ufc-fighters-statistics.csv"
DEFAULT_OUTPUT = DATA_DIR / "ufc_fight_data.csv"


def clean_name(name: object) -> str:
    return str(name).lower().strip().replace(".", "").replace("'", "")


def build_ufc_fight_data(results_path: Path, fighter_stats_path: Path) -> pd.DataFrame:
    fight_results = pd.read_csv(results_path)
    fighter_stats = pd.read_csv(fighter_stats_path)

    fighter_stats = fighter_stats.drop(columns=["nickname"], errors="ignore")
    num_cols = fighter_stats.select_dtypes("float").columns
    fighter_stats[num_cols] = fighter_stats[num_cols].fillna(fighter_stats[num_cols].median())
    fighter_stats["date_of_birth"] = pd.to_datetime(fighter_stats["date_of_birth"], errors="coerce")
    today = pd.Timestamp.today()
    fighter_stats["age"] = fighter_stats["date_of_birth"].apply(
        lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        if pd.notnull(d)
        else np.nan
    )
    fighter_stats = fighter_stats.drop(columns=["date_of_birth"])
    fighter_stats["age"] = fighter_stats["age"].fillna(fighter_stats["age"].median())
    fighter_stats["stance"] = fighter_stats["stance"].fillna("Unknown")
    fighter_stats = pd.get_dummies(fighter_stats, columns=["stance"], prefix="stance", dtype=int)
    fighter_stats["name"] = fighter_stats["name"].apply(clean_name)

    fight_results[["fighter_1", "fighter_2"]] = fight_results["BOUT"].str.split(" vs. ", expand=True)
    fight_results["fighter_1"] = fight_results["fighter_1"].apply(clean_name)
    fight_results["fighter_2"] = fight_results["fighter_2"].apply(clean_name)

    f1 = fight_results[["EVENT", "fighter_1", "OUTCOME"]].rename(
        columns={"fighter_1": "name", "OUTCOME": "outcome_raw"}
    )
    f2 = fight_results[["EVENT", "fighter_2", "OUTCOME"]].rename(
        columns={"fighter_2": "name", "OUTCOME": "outcome_raw"}
    )
    f1["win"] = (f1["outcome_raw"].str[0] == "W").astype(int)
    f2["win"] = (f2["outcome_raw"].str[-1] == "W").astype(int)
    all_fights = pd.concat([f1[["EVENT", "name", "win"]], f2[["EVENT", "name", "win"]]])
    all_fights = all_fights.sort_values("EVENT")
    all_fights["rolling_win_rate"] = all_fights.groupby("name")["win"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift()
    )
    latest_rolling = (
        all_fights.dropna(subset=["rolling_win_rate"])
        .groupby("name", as_index=False)
        .tail(1)[["name", "rolling_win_rate"]]
        .drop_duplicates("name")
    )
    fighter_stats = fighter_stats.merge(latest_rolling, on="name", how="left")
    fighter_stats["rolling_win_rate"] = fighter_stats["rolling_win_rate"].fillna(
        fighter_stats["rolling_win_rate"].mean()
    )

    decisive = fight_results[fight_results["OUTCOME"].isin(["W/L", "L/W"])].copy()
    decisive["outcome"] = (decisive["OUTCOME"] == "W/L").astype(int)
    fight_data = decisive[["EVENT", "fighter_1", "fighter_2", "outcome"]].merge(
        fighter_stats, left_on="fighter_1", right_on="name", how="inner"
    ).merge(
        fighter_stats,
        left_on="fighter_2",
        right_on="name",
        how="inner",
        suffixes=("_f1", "_f2"),
    )
    fight_data = fight_data.drop(columns=["EVENT", "fighter_1", "fighter_2", "name_f1", "name_f2"]).dropna()

    stat_cols = [
        "wins",
        "losses",
        "draws",
        "height_cm",
        "weight_in_kg",
        "reach_in_cm",
        "significant_strikes_landed_per_minute",
        "significant_striking_accuracy",
        "significant_strikes_absorbed_per_minute",
        "significant_strike_defence",
        "average_takedowns_landed_per_15_minutes",
        "takedown_accuracy",
        "takedown_defense",
        "average_submissions_attempted_per_15_minutes",
        "age",
        "rolling_win_rate",
    ]
    for col in stat_cols:
        if f"{col}_f1" in fight_data.columns:
            fight_data[f"diff_{col}"] = fight_data[f"{col}_f1"] - fight_data[f"{col}_f2"]

    return fight_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update ufc_fight_data.csv.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTER_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-ufcstats", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.refresh_ufcstats or not args.results.exists():
        update_ufcstats_fight_data(dry_run=args.dry_run, backup=not args.no_backup)

    df = build_ufc_fight_data(args.results, args.fighter_stats)
    print(f"Built {args.output} with shape {df.shape}")
    if not args.dry_run:
        write_csv_atomic(df, args.output, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
