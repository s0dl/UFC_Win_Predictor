# UFC Win Predictor

UFC fight prediction app with a React frontend and FastAPI/XGBoost backend.

## Run Locally

Start the combined app container:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8080
```

Health check:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/health
```

## Build Production Image

```bash
docker build -f Dockerfile -t ufc-win-predictor:local .
```

## Deploy

Terraform for Cloud Run lives in `infra/terraform`. It deploys a single Cloud
Run service from the combined app image.
