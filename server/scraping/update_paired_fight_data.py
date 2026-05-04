"""Update server/models/data/paired_fight_data.csv for the XGBoost pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scraping_common import DATA_DIR, write_csv_atomic
from update_ufc_fight_results_with_odds import DEFAULT_OUTPUT as DEFAULT_RESULTS_WITH_ODDS


DEFAULT_STATS = DATA_DIR / "ufc_fight_stats.csv"
DEFAULT_FIGHTER_STATS = DATA_DIR / "ufc-fighters-statistics.csv"
DEFAULT_OUTPUT = DATA_DIR / "paired_fight_data.csv"


def clean_name(value: object) -> str:
    return str(value).lower().strip().replace(".", "").replace("'", "")


def parse_of(series: pd.Series, landed_name: str, att_name: str) -> tuple[pd.Series, pd.Series]:
    landed = series.str.extract(r"(\d+)\s+of\s+\d+")[0].astype(float)
    attempted = series.str.extract(r"\d+\s+of\s+(\d+)")[0].astype(float)
    return landed.rename(landed_name), attempted.rename(att_name)


def ctrl_to_sec(value: object) -> int:
    try:
        minutes, seconds = str(value).split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return 0


def safe_div(left: pd.Series, right: pd.Series) -> np.ndarray:
    return np.where(right > 0, left / right, 0.0)


def build_paired_fight_data(stats_path: Path, results_path: Path, fighter_stats_path: Path) -> pd.DataFrame:
    stats = pd.read_csv(stats_path).dropna(subset=["FIGHTER"])
    stats["EVENT"] = stats["EVENT"].str.strip()
    stats["BOUT"] = stats["BOUT"].str.strip()

    of_columns = {
        "SIG.STR.": ("sig_landed", "sig_att"),
        "TOTAL STR.": ("total_landed", "total_att"),
        "TD": ("td_landed", "td_att"),
        "HEAD": ("head_landed", "head_att"),
        "BODY": ("body_landed", "body_att"),
        "LEG": ("leg_landed", "leg_att"),
        "DISTANCE": ("dist_landed", "dist_att"),
        "CLINCH": ("clinch_landed", "clinch_att"),
        "GROUND": ("ground_landed", "ground_att"),
    }
    for col, (landed, attempted) in of_columns.items():
        stats[landed], stats[attempted] = parse_of(stats[col], landed, attempted)

    stats["ctrl_sec"] = stats["CTRL"].apply(ctrl_to_sec)
    for col in ["KD", "SUB.ATT", "REV."]:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)
    stats["FIGHTER"] = stats["FIGHTER"].apply(clean_name)

    agg_cols = [
        "sig_landed",
        "sig_att",
        "total_landed",
        "total_att",
        "td_landed",
        "td_att",
        "head_landed",
        "head_att",
        "body_landed",
        "body_att",
        "leg_landed",
        "leg_att",
        "dist_landed",
        "dist_att",
        "clinch_landed",
        "clinch_att",
        "ground_landed",
        "ground_att",
        "ctrl_sec",
        "KD",
        "SUB.ATT",
        "REV.",
    ]
    fight_level = stats.groupby(["EVENT", "BOUT", "FIGHTER"])[agg_cols].sum().reset_index()
    fight_level["sig_acc"] = safe_div(fight_level["sig_landed"], fight_level["sig_att"])
    fight_level["td_acc"] = safe_div(fight_level["td_landed"], fight_level["td_att"])
    fight_level["head_rate"] = safe_div(fight_level["head_landed"], fight_level["sig_landed"])
    fight_level["body_rate"] = safe_div(fight_level["body_landed"], fight_level["sig_landed"])
    fight_level["leg_rate"] = safe_div(fight_level["leg_landed"], fight_level["sig_landed"])
    fight_level["dist_rate"] = safe_div(fight_level["dist_landed"], fight_level["total_landed"])
    fight_level["clinch_rate"] = safe_div(fight_level["clinch_landed"], fight_level["total_landed"])
    fight_level["ground_rate"] = safe_div(fight_level["ground_landed"], fight_level["total_landed"])

    results = pd.read_csv(results_path)
    results["EVENT"] = results["EVENT"].str.strip()
    results["BOUT"] = results["BOUT"].str.strip()
    results[["f1", "f2"]] = results["BOUT"].str.split(" vs. ", expand=True)
    results["f1"] = results["f1"].str.strip().apply(clean_name)
    results["f2"] = results["f2"].str.strip().apply(clean_name)

    odds_cols = ["f1_open_odds", "f2_open_odds", "f1_close_odds", "f2_close_odds"]
    for col in odds_cols:
        if col not in results.columns:
            results[col] = np.nan
        results[col] = pd.to_numeric(results[col], errors="coerce")

    if "odds_inferred" not in results.columns:
        results["odds_inferred"] = False
    results["odds_inferred"] = results["odds_inferred"].astype(int)

    decisive = results[results["OUTCOME"].isin(["W/L", "L/W"])][
        ["EVENT", "BOUT", "f1", "f2", "OUTCOME", "odds_inferred"] + odds_cols
    ].copy()
    decisive["outcome"] = (decisive["OUTCOME"] == "W/L").astype(int)

    fl = fight_level.merge(
        decisive[["EVENT", "BOUT", "f1", "f2", "outcome", "odds_inferred"] + odds_cols],
        on=["EVENT", "BOUT"],
        how="inner",
    )
    fl["is_f1"] = fl["FIGHTER"] == fl["f1"]
    fl["won"] = np.where(fl["is_f1"], fl["outcome"], 1 - fl["outcome"])
    fl["opponent"] = np.where(fl["is_f1"], fl["f2"], fl["f1"])
    fl["open_odds"] = np.where(fl["is_f1"], fl["f1_open_odds"], fl["f2_open_odds"])
    fl["close_odds"] = np.where(fl["is_f1"], fl["f1_close_odds"], fl["f2_close_odds"])

    event_order = results[["EVENT"]].drop_duplicates().reset_index(drop=True)
    event_order["event_idx"] = event_order.index.max() - event_order.index
    fl = fl.merge(event_order, on="EVENT", how="left")
    fl = fl.sort_values(["FIGHTER", "event_idx"]).reset_index(drop=True)

    roll_base = [
        "sig_landed",
        "sig_att",
        "sig_acc",
        "td_landed",
        "td_att",
        "td_acc",
        "head_rate",
        "body_rate",
        "leg_rate",
        "dist_rate",
        "clinch_rate",
        "ground_rate",
        "ctrl_sec",
        "KD",
        "SUB.ATT",
        "REV.",
        "won",
    ]
    for window in [3, 5]:
        for feature in roll_base:
            fl[f"roll{window}_{feature}"] = fl.groupby("FIGHTER")[feature].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    fl["n_prior_fights"] = fl.groupby("FIGHTER").cumcount()
    fl_model = fl[fl["n_prior_fights"] >= 1].copy()
    roll_cols = [col for col in fl_model.columns if col.startswith("roll")]

    fs = pd.read_csv(fighter_stats_path).drop(columns=["nickname"], errors="ignore")
    num_cols = fs.select_dtypes("float").columns
    fs[num_cols] = fs[num_cols].fillna(fs[num_cols].median())
    fs["date_of_birth"] = pd.to_datetime(fs["date_of_birth"], errors="coerce")
    today = pd.Timestamp.today()
    fs["age"] = fs["date_of_birth"].apply(
        lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        if pd.notnull(d)
        else np.nan
    )
    fs = fs.drop(columns=["date_of_birth"])
    fs["age"] = fs["age"].fillna(fs["age"].median())
    fs["stance"] = fs["stance"].fillna("Unknown")
    fs = pd.get_dummies(fs, columns=["stance"], prefix="stance", dtype=int)
    fs["name"] = fs["name"].apply(clean_name)

    career_cols = [col for col in fs.columns if col != "name"]
    current_odds_cols = ["open_odds", "close_odds"]
    fight_level_cols = ["odds_inferred"]
    fighter_view = fl_model[
        ["EVENT", "BOUT", "FIGHTER", "opponent", "won", "event_idx"]
        + roll_cols
        + current_odds_cols
        + fight_level_cols
    ].copy()

    side_specific_cols = roll_cols + current_odds_cols
    opp_view = fighter_view[["EVENT", "BOUT", "FIGHTER"] + side_specific_cols].copy()
    opp_view = opp_view.rename(columns={col: f"{col}_opp" for col in side_specific_cols})
    opp_view = opp_view.rename(columns={"FIGHTER": "opponent"})

    self_view = fighter_view.rename(columns={col: f"{col}_self" for col in side_specific_cols})
    paired = self_view.merge(opp_view, on=["EVENT", "BOUT", "opponent"], how="inner")
    paired = paired.merge(
        fs.rename(columns={col: f"career_{col}" for col in career_cols} | {"name": "name"}),
        left_on="FIGHTER",
        right_on="name",
        how="left",
    ).drop(columns=["name"])
    paired = paired.merge(
        fs.rename(columns={col: f"opp_career_{col}" for col in career_cols} | {"name": "name"}),
        left_on="opponent",
        right_on="name",
        how="left",
    ).drop(columns=["name"])

    paired = paired.dropna(
        subset=[
            col
            for col in paired.columns
            if col not in [f"{odds}_self" for odds in current_odds_cols]
            + [f"{odds}_opp" for odds in current_odds_cols]
        ]
    )

    career_self = [col for col in paired.columns if col.startswith("career_")]
    career_opp = [col for col in paired.columns if col.startswith("opp_career_")]
    for col in side_specific_cols:
        paired[f"diff_{col}"] = paired[f"{col}_self"] - paired[f"{col}_opp"]
    for self_col, opp_col in zip(career_self, career_opp):
        paired[f"diff_{self_col[7:]}"] = paired[self_col] - paired[opp_col]

    return paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update paired_fight_data.csv.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_WITH_ODDS)
    parser.add_argument("--fighter-stats", type=Path, default=DEFAULT_FIGHTER_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = build_paired_fight_data(args.stats, args.results, args.fighter_stats)
    print(f"Built {args.output} with shape {df.shape}")
    if not args.dry_run:
        write_csv_atomic(df, args.output, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
