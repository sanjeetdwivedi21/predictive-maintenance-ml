# Predictive Maintenance ML — Vercel

## Required model artifacts
Place these files in the repository root before deploying:
- `pm_meta.joblib`
- `pm_xgb.json`

The metadata artifact must contain `features`, `types`, `threshold`, and `type_model`, as expected by the original app.

## Deploy
1. Commit and push this project to GitHub.
2. Import the repository in Vercel.
3. Framework Preset: Other; Root Directory: `./`.
4. Leave build/output/install overrides disabled and deploy.

The frontend is `index.html`; the Python API is `api/index.py`. The endpoint is `POST /api/predict`; health check is `GET /api/health`.

This is a demonstration model trained on AI4I 2020 benchmark data, not a validated industrial safety system.
