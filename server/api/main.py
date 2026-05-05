from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


SERVER_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = SERVER_ROOT / "static"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_EXEMPT_PATHS = {"/", "/health", "/api/health"}
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from models.ufc_xgboost import get_predictor


app = FastAPI(title="UFC Fight Predictor")
request_log: defaultdict[str, deque[float]] = defaultdict(deque)


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
