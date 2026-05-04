# UFC Prediction Server

This folder contains the backend-ready project layout.

## Layout

- `api/` - FastAPI entrypoint for frontend calls.
- `models/` - model code, notebooks, training scripts, artifacts, data, and reports.
- `scraping/` - UFCStats and BestFightOdds scraping code and scraped outputs.
- `original_repo/` - legacy/original project files kept for reference.

## Run API

From the repository root:

```bash
conda run -n ufc uvicorn server.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fighter1":"Islam Makhachev","fighter2":"Charles Oliveira"}'
```

You can pass odds when known:

```json
{
  "fighter1": "Islam Makhachev",
  "fighter2": "Charles Oliveira",
  "fighter1_open_odds": -250,
  "fighter2_open_odds": 210,
  "fighter1_close_odds": -300,
  "fighter2_close_odds": 240,
  "odds_inferred": 0
}
```

## Update Fighter Statistics

Refresh `models/data/ufc-fighters-statistics.csv` from UFCStats:

```bash
conda run -n ufc python server/scraping/update_fighter_statistics.py
```

For a faster append-only update that scrapes only newly discovered fighter names:

```bash
conda run -n ufc python server/scraping/update_fighter_statistics.py --only-new
```

The updater writes a `.bak` copy before replacing the CSV. Use `--dry-run` to
discover and scrape without writing files.
