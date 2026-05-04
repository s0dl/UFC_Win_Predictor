from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from models.ufc_xgboost import get_predictor


app = FastAPI(title="UFC Fight Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    fighter1: str
    fighter2: str
    fighter1_open_odds: float | None = None
    fighter2_open_odds: float | None = None
    fighter1_close_odds: float | None = None
    fighter2_close_odds: float | None = None
    odds_inferred: int = 1


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    try:
        result = get_predictor().predict(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.as_dict()


@app.get("/fighters")
def fighters() -> dict:
    return {"fighters": get_predictor().list_fighters()}


@app.get("/rankings")
def rankings(limit: int = 5000, benchmark_count: int = 10) -> dict:
    benchmark_count = min(max(benchmark_count, 5), 100)
    limit = min(max(limit, 1), 5000)
    ranked = get_predictor().rank_fighters(benchmark_count=benchmark_count)
    return {
        "benchmark_count": benchmark_count,
        "total": len(ranked),
        "rankings": [item.as_dict() for item in ranked[:limit]],
    }
