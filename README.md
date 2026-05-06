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
- Optional opening and closing moneyline inputs
- FastAPI inference API backed by saved XGBoost artifacts
- Scraper/update scripts for UFCStats and BestFightOdds-derived CSVs
- Dockerized single-container app serving frontend and backend together
- Terraform configuration for Google Cloud Run
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
python server/scraping/update_fighter_statistics.py
python server/scraping/update_ufc_fight_stats.py
python server/scraping/update_ufc_fight_results_with_odds.py
python server/scraping/update_ufc_fight_data.py
python server/scraping/update_paired_fight_data.py
```

## Deployment

Build and push the production image:

```bash
PROJECT_ID=your-gcp-project-id
REGION=us-central1
REPO=$REGION-docker.pkg.dev/$PROJECT_ID/ufc

docker build -f Dockerfile -t $REPO/ufc-app .
docker push $REPO/ufc-app
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

## Model Notes

The production API loads saved XGBoost artifacts from
`server/models/artifacts`. Predictions use fighter career features, recent fight
history, rolling performance stats, and optional moneyline odds.

This is a sports prediction tool, not betting advice. Model output should be
treated as an estimate based on available historical data and feature quality.

## Security And Operations

- Single public Cloud Run service reduces cross-service auth complexity
- API is same-origin under `/api`
- Rate limiting is enabled in FastAPI
- CORS is disabled unless `CORS_ALLOW_ORIGINS` is set for local development
- Cloud Run scaling limits are managed through Terraform

## Future Improvements

- Add automated API/model tests
- Add CI for frontend build, backend compile, and Docker build
- Add model metrics and calibration summary to this README
- Add screenshot or GIF of the deployed app
- Add custom domain once the demo URL is finalized
