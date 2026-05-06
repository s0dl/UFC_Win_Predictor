from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"


def clean_name(name: str) -> str:
    return str(name).lower().strip().replace(".", "").replace("'", "")


@dataclass
class PredictionResult:
    fighter1: str
    fighter2: str
    fighter1_win_probability: float
    fighter2_win_probability: float
    predicted_winner: str
    predicted_winner_probability: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "fighter1": self.fighter1,
            "fighter2": self.fighter2,
            "fighter1_win_probability": self.fighter1_win_probability,
            "fighter2_win_probability": self.fighter2_win_probability,
            "predicted_winner": self.predicted_winner,
            "predicted_winner_probability": self.predicted_winner_probability,
        }


@dataclass
class FighterRanking:
    rank: int
    fighter: str
    model_score: float
    benchmark_wins: int
    benchmark_count: int
    prior_fights: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "fighter": self.fighter,
            "model_score": self.model_score,
            "benchmark_wins": self.benchmark_wins,
            "benchmark_count": self.benchmark_count,
            "prior_fights": self.prior_fights,
        }


def display_name(name: str) -> str:
    return " ".join(part.capitalize() for part in str(name).split())


class UFCXGBoostPredictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.model = xgb.XGBClassifier()
        self.model.load_model(artifact_dir / "xgb_fight_model.json")

        with open(artifact_dir / "paired_feature_cols.pkl", "rb") as handle:
            self.feature_cols: list[str] = pickle.load(handle)
        with open(artifact_dir / "fighter_stats_clean.pkl", "rb") as handle:
            self.fighter_stats: pd.DataFrame = pickle.load(handle)
        with open(artifact_dir / "fl_model_final.pkl", "rb") as handle:
            self.fight_history: pd.DataFrame = pickle.load(handle)

        self.fighter_stats = self._load_current_fighter_stats(self.fighter_stats)
        self.roll_cols = [col for col in self.fight_history.columns if col.startswith("roll")]
        self.career_cols = [col for col in self.fighter_stats.columns if col != "name"]
        self.known_fighters = set(self.fight_history["FIGHTER"].dropna().astype(str))
        self.known_career_fighters = set(self.fighter_stats["name"].dropna().astype(str))
        self.available_fighters = sorted(self.known_career_fighters)
        self.rankable_fighters = sorted(self.known_fighters & self.known_career_fighters)
        self.fight_counts = self.fight_history.groupby("FIGHTER").size().to_dict()
        self.display_fight_counts = self._load_current_fight_counts(self.fight_counts)
        self.latest_history = (
            self.fight_history.dropna(subset=["FIGHTER"])
            .groupby("FIGHTER", as_index=False)
            .tail(1)
            .set_index("FIGHTER")
        )
        self.career_by_name = self.fighter_stats.dropna(subset=["name"]).set_index("name")

    def _load_current_fighter_stats(self, fallback: pd.DataFrame) -> pd.DataFrame:
        csv_path = ROOT / "models" / "data" / "ufc-fighters-statistics.csv"
        if not csv_path.exists():
            return fallback

        try:
            stats = pd.read_csv(csv_path).drop(columns=["nickname"], errors="ignore")
        except (OSError, pd.errors.ParserError):
            return fallback

        num_cols = stats.select_dtypes("float").columns
        stats[num_cols] = stats[num_cols].fillna(stats[num_cols].median())
        stats["date_of_birth"] = pd.to_datetime(stats["date_of_birth"], errors="coerce")
        today = pd.Timestamp.today()
        stats["age"] = stats["date_of_birth"].apply(
            lambda date: today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            if pd.notnull(date)
            else np.nan
        )
        stats = stats.drop(columns=["date_of_birth"])
        stats["age"] = stats["age"].fillna(stats["age"].median())
        stats["stance"] = stats["stance"].fillna("Unknown")
        stats = pd.get_dummies(stats, columns=["stance"], prefix="stance", dtype=int)
        stats["name"] = stats["name"].apply(clean_name)
        return stats.reindex(columns=fallback.columns, fill_value=0)

    def _load_current_fight_counts(self, fallback: dict[str, int]) -> dict[str, int]:
        csv_path = ROOT / "models" / "data" / "ufc_fight_stats.csv"
        if not csv_path.exists():
            return fallback

        try:
            stats = pd.read_csv(csv_path, usecols=["EVENT", "BOUT", "FIGHTER"]).dropna(subset=["FIGHTER"])
        except (OSError, ValueError, pd.errors.ParserError):
            return fallback

        stats["FIGHTER"] = stats["FIGHTER"].apply(clean_name)
        counts = (
            stats.drop_duplicates(["EVENT", "BOUT", "FIGHTER"])
            .groupby("FIGHTER")
            .size()
            .astype(int)
            .to_dict()
        )
        return {**fallback, **counts}

    def _build_row(
        self,
        fighter_name: str,
        opponent_name: str,
        open_odds: float | None = None,
        close_odds: float | None = None,
        opponent_open_odds: float | None = None,
        opponent_close_odds: float | None = None,
        odds_inferred: int = 1,
    ) -> pd.DataFrame:
        fighter = clean_name(fighter_name)
        opponent = clean_name(opponent_name)

        if fighter not in self.known_career_fighters:
            raise ValueError(f"{fighter_name!r} was not found in fighter stats")
        if opponent not in self.known_career_fighters:
            raise ValueError(f"{opponent_name!r} was not found in fighter stats")

        fighter_row = (
            self.fight_history[self.fight_history["FIGHTER"] == fighter].iloc[-1]
            if fighter in self.known_fighters
            else None
        )
        if opponent in self.known_fighters:
            opponent_row = self.fight_history[self.fight_history["FIGHTER"] == opponent].iloc[-1]
            opponent_rolling = {f"{col}_opp": opponent_row[col] for col in self.roll_cols}
        else:
            opponent_rolling = {f"{col}_opp": 0.0 for col in self.roll_cols}

        career_self = self.fighter_stats[self.fighter_stats["name"] == fighter].iloc[0]
        career_opp = self.fighter_stats[self.fighter_stats["name"] == opponent].iloc[0]

        row: dict[str, Any] = {}
        for col in self.roll_cols:
            row[f"{col}_self"] = fighter_row[col] if fighter_row is not None else 0.0
        row["open_odds_self"] = np.nan if open_odds is None else open_odds
        row["close_odds_self"] = np.nan if close_odds is None else close_odds
        row.update(opponent_rolling)
        row["open_odds_opp"] = np.nan if opponent_open_odds is None else opponent_open_odds
        row["close_odds_opp"] = np.nan if opponent_close_odds is None else opponent_close_odds
        row["odds_inferred"] = odds_inferred

        for col in self.career_cols:
            row[f"career_{col}"] = career_self[col]
            row[f"opp_career_{col}"] = career_opp[col]

        for col in self.roll_cols + ["open_odds", "close_odds"]:
            row[f"diff_{col}"] = row[f"{col}_self"] - row[f"{col}_opp"]
        for col in self.career_cols:
            row[f"diff_{col}"] = row[f"career_{col}"] - row[f"opp_career_{col}"]

        return pd.DataFrame([row]).reindex(columns=self.feature_cols)

    def predict(
        self,
        fighter1: str,
        fighter2: str,
        fighter1_open_odds: float | None = None,
        fighter2_open_odds: float | None = None,
        fighter1_close_odds: float | None = None,
        fighter2_close_odds: float | None = None,
        odds_inferred: int = 1,
    ) -> PredictionResult:
        row_ab = self._build_row(
            fighter1,
            fighter2,
            fighter1_open_odds,
            fighter1_close_odds,
            fighter2_open_odds,
            fighter2_close_odds,
            odds_inferred,
        )
        row_ba = self._build_row(
            fighter2,
            fighter1,
            fighter2_open_odds,
            fighter2_close_odds,
            fighter1_open_odds,
            fighter1_close_odds,
            odds_inferred,
        )

        p_ab = float(self.model.predict_proba(row_ab)[0, 1])
        p_ba = float(self.model.predict_proba(row_ba)[0, 1])
        fighter1_probability = (p_ab + (1.0 - p_ba)) / 2.0
        fighter2_probability = 1.0 - fighter1_probability
        if fighter1_probability >= fighter2_probability:
            winner = fighter1
            winner_probability = fighter1_probability
        else:
            winner = fighter2
            winner_probability = fighter2_probability
        return PredictionResult(
            fighter1=fighter1,
            fighter2=fighter2,
            fighter1_win_probability=fighter1_probability,
            fighter2_win_probability=fighter2_probability,
            predicted_winner=winner,
            predicted_winner_probability=winner_probability,
        )

    def resolve_fighter_name(self, name: str, cutoff: float = 0.84) -> str:
        fighter = clean_name(name)
        if fighter in self.available_fighters:
            return display_name(fighter)

        matches = get_close_matches(fighter, self.available_fighters, n=1, cutoff=cutoff)
        if not matches:
            raise ValueError(f"{name!r} was not found in model fighter data")
        return display_name(matches[0])

    def list_fighters(self) -> list[dict[str, Any]]:
        return [
            {
                "name": display_name(fighter),
                "key": fighter,
                "prior_fights": int(self.display_fight_counts.get(fighter, 0)),
            }
            for fighter in self.available_fighters
        ]

    def _benchmark_fighters(self, benchmark_count: int) -> list[str]:
        ranked_by_experience = sorted(
            self.rankable_fighters,
            key=lambda fighter: (self.fight_counts.get(fighter, 0), fighter),
            reverse=True,
        )
        return ranked_by_experience[:benchmark_count]

    def _ranking_feature_row(self, fighter: str, opponent: str) -> dict[str, Any]:
        fighter_row = self.latest_history.loc[fighter]
        opponent_row = self.latest_history.loc[opponent]
        career_self = self.career_by_name.loc[fighter]
        career_opp = self.career_by_name.loc[opponent]

        row: dict[str, Any] = {}
        for col in self.roll_cols:
            row[f"{col}_self"] = fighter_row[col]
            row[f"{col}_opp"] = opponent_row[col]
            row[f"diff_{col}"] = fighter_row[col] - opponent_row[col]

        row["open_odds_self"] = np.nan
        row["close_odds_self"] = np.nan
        row["open_odds_opp"] = np.nan
        row["close_odds_opp"] = np.nan
        row["diff_open_odds"] = np.nan
        row["diff_close_odds"] = np.nan
        row["odds_inferred"] = 1

        for col in self.career_cols:
            row[f"career_{col}"] = career_self[col]
            row[f"opp_career_{col}"] = career_opp[col]
            row[f"diff_{col}"] = career_self[col] - career_opp[col]

        return row

    @lru_cache(maxsize=8)
    def rank_fighters(self, benchmark_count: int = 40) -> tuple[FighterRanking, ...]:
        benchmarks = self._benchmark_fighters(benchmark_count)
        rows = []
        row_fighters = []
        for fighter in self.rankable_fighters:
            for opponent in benchmarks:
                if opponent == fighter:
                    continue
                rows.append(self._ranking_feature_row(fighter, opponent))
                row_fighters.append(fighter)

        if not rows:
            return tuple()

        feature_frame = pd.DataFrame(rows).reindex(columns=self.feature_cols)
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        probabilities = self.model.predict_proba(feature_frame)[:, 1]
        probability_frame = pd.DataFrame({"fighter": row_fighters, "probability": probabilities})

        rankings = []
        for fighter, group in probability_frame.groupby("fighter"):
            if group.empty:
                continue
            model_score = float(group["probability"].mean())
            rankings.append(
                {
                    "fighter": fighter,
                    "model_score": model_score,
                    "benchmark_wins": int((group["probability"] >= 0.5).sum()),
                    "benchmark_count": int(len(group)),
                    "prior_fights": int(self.fight_counts.get(fighter, 0)),
                }
            )

        rankings = sorted(
            rankings,
            key=lambda item: (item["model_score"], item["benchmark_wins"], item["prior_fights"]),
            reverse=True,
        )
        return tuple(
            FighterRanking(
                rank=index,
                fighter=display_name(item["fighter"]),
                model_score=item["model_score"],
                benchmark_wins=item["benchmark_wins"],
                benchmark_count=item["benchmark_count"],
                prior_fights=item["prior_fights"],
            )
            for index, item in enumerate(rankings, start=1)
        )


_predictor: UFCXGBoostPredictor | None = None


def get_predictor() -> UFCXGBoostPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UFCXGBoostPredictor()
    return _predictor
