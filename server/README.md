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
uvicorn server.api.main:app --reload --host 0.0.0.0 --port 8000
```

If running the Vite frontend separately, allow that local origin:

```bash
CORS_ALLOW_ORIGINS=http://localhost:5173 uvicorn server.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run With Docker

From the repository root:

```bash
docker compose up --build
```

Open the frontend at:

```text
http://localhost:8080
```

The API is available from the same container:

```text
http://localhost:8080/api
```

Prediction:

```bash
curl -X POST http://localhost:8080/api/predict \
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

## Update Data CSVs

Refresh `models/data/ufc-fighters-statistics.csv` from UFCStats:

```bash
conda run -n ufc python server/scraping/update_fighter_statistics.py
```

For a faster append-only update that scrapes only newly discovered fighter names:

```bash
python server/scraping/update_fighter_statistics.py --only-new
```

The updater writes a `.bak` copy before replacing the CSV. Use `--dry-run` to
discover and scrape without writing files.

Refresh the other current data CSVs:

```bash
python server/scraping/update_ufc_fight_stats.py
python server/scraping/update_ufc_fight_results_with_odds.py
python server/scraping/update_ufc_fight_data.py
python server/scraping/update_paired_fight_data.py
```

Scraper intermediates live in `scraping/cache/`. `update_ufc_fight_stats.py`
maintains the UFCStats cache files needed by the odds join.
`update_ufc_fight_results_with_odds.py` uses the existing BestFightOdds
moneyline cache by default; pass `--refresh-odds` to re-scrape it.
