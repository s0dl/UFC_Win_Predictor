from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from collections import defaultdict, deque
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


SERVER_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = SERVER_ROOT / "static"
NEXT_EVENT_CARD_PATH = SERVER_ROOT / "models" / "data" / "ufc_next_event_card.csv"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_EXEMPT_PATHS = {"/", "/health", "/api/health"}
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from models.ufc_xgboost import get_predictor


app = FastAPI(title="UFC Fight Predictor")
request_log: defaultdict[str, deque[float]] = defaultdict(deque)

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in RATE_LIMIT_EXEMPT_PATHS or request.url.path.startswith("/assets/"):
        return await call_next(request)

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    key = client_key(request)
    timestamps = request_log[key]

    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    timestamps.append(now)
    return await call_next(request)


class PredictionRequest(BaseModel):
    fighter1: str
    fighter2: str
    fighter1_open_odds: float | None = None
    fighter2_open_odds: float | None = None
    fighter1_close_odds: float | None = None
    fighter2_close_odds: float | None = None
    odds_inferred: int = 1


def no_vig_probabilities(odds1: int | float | None, odds2: int | float | None) -> tuple[float | None, float | None]:
    if odds1 is None or odds2 is None:
        return None, None
    raw1 = american_to_implied_probability(int(odds1))
    raw2 = american_to_implied_probability(int(odds2))
    total = raw1 + raw2
    if total <= 0:
        return None, None
    return raw1 / total, raw2 / total


def edge_payload(model_probability: float, implied_probability: float | None) -> dict[str, Any]:
    if implied_probability is None:
        return {
            "implied_probability_no_vig": None,
            "edge": None,
            "kelly_fraction": None,
            "has_edge": False,
        }

    edge = model_probability - implied_probability
    denominator = 1.0 - implied_probability
    kelly_fraction = edge / denominator if denominator > 0 else None
    return {
        "implied_probability_no_vig": implied_probability,
        "edge": edge,
        "kelly_fraction": kelly_fraction,
        "has_edge": edge >= 0.05,
    }


def comparable_edge(edge: float | None) -> float:
    return edge if edge is not None else -1.0


def american_to_implied_probability(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def clean_cached_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def cached_next_event_card(path: Path | None = None) -> dict[str, Any]:
    path = path or NEXT_EVENT_CARD_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run `python server/scraping/update_next_ufc_event_card.py` first."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} does not contain any cached fights.")

    df = df.sort_values("importance_order", kind="stable")
    first = df.iloc[0]
    fights = []
    for _, row in df.iterrows():
        fights.append(
            {
                column: clean_cached_value(row[column])
                for column in df.columns
                if column not in {"event", "date", "source_url", "scraped_at"}
            }
        )

    return {
        "event": clean_cached_value(first.get("event")),
        "date": clean_cached_value(first.get("date")),
        "source_url": clean_cached_value(first.get("source_url")),
        "scraped_at": clean_cached_value(first.get("scraped_at")),
        "fights": fights,
    }


def upcoming_event_edges() -> dict[str, Any]:
    card = cached_next_event_card()
    predictor = get_predictor()
    enriched_fights = []

    for fight in card["fights"]:
        try:
            fighter1 = predictor.resolve_fighter_name(str(fight["fighter1"]))
            fighter2 = predictor.resolve_fighter_name(str(fight["fighter2"]))
            current1 = fight.get("fighter1_current_odds")
            current2 = fight.get("fighter2_current_odds")
            open1 = fight.get("fighter1_open_odds")
            open2 = fight.get("fighter2_open_odds")
            prediction = predictor.predict(
                fighter1=fighter1,
                fighter2=fighter2,
                fighter1_open_odds=open1,
                fighter2_open_odds=open2,
                fighter1_close_odds=current1,
                fighter2_close_odds=current2,
                odds_inferred=0 if current1 is not None and current2 is not None else 1,
            ).as_dict()
            implied1, implied2 = no_vig_probabilities(current1, current2)
            fighter1_edge = edge_payload(prediction["fighter1_win_probability"], implied1)
            fighter2_edge = edge_payload(prediction["fighter2_win_probability"], implied2)
            recommended_side = (
                fighter1
                if comparable_edge(fighter1_edge["edge"]) >= comparable_edge(fighter2_edge["edge"])
                else fighter2
            )
            recommended_edge = fighter1_edge if recommended_side == fighter1 else fighter2_edge
            enriched_fights.append(
                {
                    **fight,
                    "fighter1": fighter1,
                    "fighter2": fighter2,
                    "prediction": prediction,
                    "fighter1_edge": fighter1_edge,
                    "fighter2_edge": fighter2_edge,
                    "recommended_side": recommended_side,
                    "recommended_edge": recommended_edge,
                    "model_error": None,
                }
            )
        except ValueError as exc:
            enriched_fights.append({**fight, "prediction": None, "model_error": str(exc)})

    payload = {**card, "fights": enriched_fights, "edge_threshold": 0.05}
    return payload


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
@app.post("/api/predict")
def predict(payload: PredictionRequest) -> dict:
    try:
        result = get_predictor().predict(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.as_dict()


@app.get("/fighters")
@app.get("/api/fighters")
def fighters() -> dict:
    return {"fighters": get_predictor().list_fighters()}


@app.get("/rankings")
@app.get("/api/rankings")
def rankings(limit: int = 5000, benchmark_count: int = 10) -> dict:
    benchmark_count = min(max(benchmark_count, 5), 100)
    limit = min(max(limit, 1), 5000)
    ranked = get_predictor().rank_fighters(benchmark_count=benchmark_count)
    return {
        "benchmark_count": benchmark_count,
        "total": len(ranked),
        "rankings": [item.as_dict() for item in ranked[:limit]],
    }


@app.get("/next-event/edges")
@app.get("/api/next-event/edges")
def next_event_edges() -> dict:
    try:
        return upcoming_event_edges()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if (STATIC_ROOT / "assets").exists():
    app.mount("/assets", StaticFiles(directory=STATIC_ROOT / "assets"), name="assets")


@app.get("/")
@app.get("/{path:path}")
def frontend(path: str = "") -> FileResponse:
    index_path = STATIC_ROOT / "index.html"
    requested_path = STATIC_ROOT / path

    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    if requested_path.is_file():
        return FileResponse(requested_path)
    if index_path.exists():
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Frontend build not found")
