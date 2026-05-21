# UFC Win Predictor

An end-to-end UFC fight prediction app that combines scraped fight data, XGBoost
modeling, a FastAPI inference service, a React frontend, and Google Cloud Run
infrastructure.

> Demo: ufcfightpredictor.com

## Overview

UFC Win Predictor estimates the winner of a matchup from fighter names and
optional betting odds. The app includes a searchable fighter input, probability
breakdown, API-backed inference, update scripts for the current datasets, and a
single-container production deployment.

## Features

- Searchable fighter dropdown powered by the model's known fighter database
- Fight winner prediction with per-fighter win probabilities
- Optional opening and current moneyline inputs
- Next-event edge table comparing model probability with no-vig market prices
- FastAPI inference API backed by saved XGBoost artifacts
- Scraper/update scripts for UFCStats and BestFightOdds-derived CSVs
- Dockerized single-container app serving frontend and backend together
- Terraform configuration for Google Cloud Run
- GitHub Actions CI for backend tests, frontend builds, and Docker builds
- Basic per-IP rate limiting for API routes

## Stack

- **Frontend:** React, Vite, CSS
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Modeling:** XGBoost, pandas, scikit-learn
- **Data:** UFCStats and BestFightOdds-derived CSV pipelines
- **Infra:** Docker, Docker Compose, Terraform, Google Cloud Run, Artifact Registry

## Architecture

```text
React/Vite frontend
        |
        | same-origin /api requests
        v
FastAPI app
        |
        v
XGBoost model artifacts
        |
        v
Predicted winner + probabilities
```

Production runs as one container:

```text
Cloud Run service
  └─ FastAPI serves:
      ├─ static frontend files
      ├─ /api/predict
      ├─ /api/fighters
      └─ /api/health
```

## Repository Layout

```text
frontend/          React/Vite frontend source
server/api/        FastAPI application
server/models/     model code, artifacts, datasets, training scripts
server/scraping/   update scripts and scraper helpers
infra/terraform/   Cloud Run infrastructure
Dockerfile         production single-container build
docker-compose.yml local container run
```

## Run Locally

From the repository root:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8080
```

Health checks:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/health
```

Prediction example:

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{"fighter1":"Islam Makhachev","fighter2":"Charles Oliveira"}'
```

## Development

Run the backend directly:

```bash
uvicorn server.api.main:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend separately:

```bash
cd frontend
VITE_API_URL=http://localhost:8000 npm run dev
```

If using separate frontend/backend dev servers, enable local CORS on the API:

```bash
CORS_ALLOW_ORIGINS=http://localhost:5173 uvicorn server.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Data Updates

Current model data lives under `server/models/data`. Scraper intermediates live
under `server/scraping/cache`.

Useful update commands:

```bash
python server/scraping/update_next_ufc_event_card.py
python server/scraping/update_fighter_statistics.py
python server/scraping/update_ufc_fight_stats.py
python server/scraping/update_ufc_fight_results_with_odds.py
python server/scraping/update_ufc_fight_data.py
python server/scraping/update_paired_fight_data.py
```

`update_next_ufc_event_card.py` writes `server/models/data/ufc_next_event_card.csv`.
The app reads that file at runtime for the next-event edge table; it does not
scrape BestFightOdds during user requests.

## Tests And CI

Run the backend checks locally:

```bash
python -m pip install -r server/requirements.txt pytest
python -m compileall server/api server/models server/scraping
pytest server/tests
```

Run the frontend build check:

```bash
cd frontend
npm ci
npm run build
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same core
checks on pushes and pull requests to `main`: backend dependency install,
backend compile, focused pytest coverage, frontend dependency install, and
frontend production build. It also verifies that the production Docker image
builds successfully.

## Deployment

Build and push the production image:

```bash
PROJECT_ID=your-gcp-project-id
REGION=us-central1
REPO=$REGION-docker.pkg.dev/$PROJECT_ID/ufc
TAG=$(date +%Y%m%d%H%M%S)

docker build -f Dockerfile -t $REPO/ufc-app:$TAG .
docker push $REPO/ufc-app:$TAG
```

Deploy with Terraform:

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Set:

```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
image      = "us-central1-docker.pkg.dev/your-gcp-project-id/ufc/ufc-app:YOUR_TAG"
```

Then:

```bash
terraform init
terraform plan
terraform apply
```

## Methodology

For a fuller write-up, see the docs folder:

- [Model report](docs/model-report.md)
- [Modeling appendix](docs/modeling.md)

### Data Sources

The project uses two public data sources:

- **UFCStats** for fight results, fight-level round stats, fighter bio data, and
  career summary statistics.
- **BestFightOdds** for opening and current/closing moneyline prices where they
  are available.

The update scripts normalize these sources into CSVs under `server/models/data`.
Historical scrape intermediates are kept under `server/scraping/cache` so odds
and UFCStats joins can be resumed or rebuilt without starting from scratch every
time.

### Feature Construction

The XGBoost model is trained from paired fighter rows. Each matchup is converted
into a fighter-vs-opponent feature vector containing:

- Fighter career attributes such as age, height, reach, stance, record, striking
  rates, takedown rates, and submission rates.
- Opponent career attributes using the same schema.
- Rolling fight-history features over recent UFC fights, including striking,
  grappling, control, knockdowns, submission attempts, reversals, and recent win
  rate.
- Difference features, such as fighter career stat minus opponent career stat.
- Opening and closing/current moneyline odds when present.
- An `odds_inferred` flag for rows where odds had to be inferred or were not
  directly matched.

For inference, the API predicts the matchup in both directions, averages the
two perspectives, and returns normalized win probabilities for each fighter.

### Current Fighter List

The searchable fighter list is sourced from the current fighter statistics CSV,
not only the training artifact. This means newly scraped fighters can appear in
the app before a full retrain. If a fighter has career stats but no usable
rolling-history row in the saved model artifacts, the API fills rolling features
with zeroes. That keeps inference available, but those predictions are less
informed than predictions for fighters with established UFC history in the
training data.

### Next Event Edge Table

The next-event table is cache-driven:

```bash
python server/scraping/update_next_ufc_event_card.py
```

That command discovers upcoming UFC cards from BestFightOdds' latest odds page
and writes `server/models/data/ufc_next_event_card.csv`. The API endpoint
`/api/next-event/edges` reads that file, runs model predictions for each fight,
and returns the card sorted by `importance_order`, where `1` is the main event.

The endpoint does not scrape external sites at request time. To update the card
or prices, rerun the update script and redeploy/restart with the refreshed CSV.

### Edge Calculation

For each fight, the API converts the current American moneylines into implied
probabilities and removes the sportsbook vig by normalizing both sides:

```text
raw_implied_a = american_to_probability(line_a)
raw_implied_b = american_to_probability(line_b)
no_vig_a = raw_implied_a / (raw_implied_a + raw_implied_b)
no_vig_b = raw_implied_b / (raw_implied_a + raw_implied_b)
```

The displayed edge is:

```text
edge = model_probability - no_vig_implied_probability
```

The app flags a side as having an edge when the edge is at least 5 percentage
points. It also reports a simple Kelly-style fraction:

```text
kelly_fraction = edge / (1 - no_vig_implied_probability)
```

This is shown as a sizing signal, not as betting advice.

### Limitations

- The model depends on data freshness. New fighters can be listed before the
  model artifacts are retrained, but predictions for those fighters may be less
  reliable.
- Public odds pages can change layout, naming, ordering, and availability. The
  cache script should be checked after major source-site changes.
- Moneyline movement is treated as a feature, but the model does not understand
  late injury news, short-notice replacements, weigh-in context, or non-quantified
  qualitative information.
- Model probabilities are estimates, not guarantees. Calibration can drift as
  the roster, rules environment, judging trends, and betting market behavior
  change.

This is a sports prediction tool, not betting advice. Model output should be
treated as an estimate based on available historical data and feature quality.

## Security And Operations

- Single public Cloud Run service reduces cross-service auth complexity
- API is same-origin under `/api`
- Rate limiting is enabled in FastAPI
- CORS is disabled unless `CORS_ALLOW_ORIGINS` is set for local development
- Cloud Run scaling limits are managed through Terraform
- The UI footer links to the methodology and states that model output is not
  betting advice.

## Future Improvements

- Add model metrics and calibration summary to this README
- Add screenshot or GIF of the deployed app
- Add end-to-end smoke tests against the Docker container
