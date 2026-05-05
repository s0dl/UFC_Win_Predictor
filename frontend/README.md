# UFC Win Predictor Frontend

React/Vite frontend for the UFC Win Predictor.

## Local Development

From this folder:

```bash
npm install
npm run dev
```

The dev server defaults to:

```text
http://localhost:5173
```

Set the backend URL with `VITE_API_URL` when needed:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Production Build

```bash
npm run build
```

The production deployment does not run a separate frontend server. The root
`Dockerfile` builds this frontend and copies `dist/` into the FastAPI image,
where the backend serves both the API and static frontend from one Cloud Run
service.

In production, the frontend uses:

```text
/api
```

as the API base path.
