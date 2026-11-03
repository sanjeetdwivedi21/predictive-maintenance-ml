# Machine Health Monitor (Vercel)

Predictive maintenance demo. A single page (`index.html`) plus two tiny Python functions.

- `POST /api/predict`: failure risk, likely cause and sensor notes for one machine
- `GET  /api/health`: health check
- `fleet.json`: precomputed fleet simulation (risk scores from the trained model)
- `lib/`: pure-Python inference (no numpy, xgboost or scikit-learn), model in `lib/model_lite.json` (1.7 MB)
- `tools/convert_model.py`: how `model_lite.json` was made from the original XGBoost + random-forest files

There is deliberately **no requirements.txt**: the functions need no third-party packages, so the
deployment is a few MB instead of hundreds.

Data: public AI4I 2020 benchmark (synthetic). Cross-validated recall 84.7%, precision 73.2%.
Not a validated industrial safety system.
