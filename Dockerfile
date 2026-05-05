FROM node:22-alpine AS frontend-build

WORKDIR /frontend

ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/server/requirements.txt

COPY server/api /app/server/api
COPY server/models/__init__.py /app/server/models/__init__.py
COPY server/models/ufc_xgboost.py /app/server/models/ufc_xgboost.py
COPY server/models/artifacts /app/server/models/artifacts
COPY --from=frontend-build /frontend/dist /app/server/static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn server.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
