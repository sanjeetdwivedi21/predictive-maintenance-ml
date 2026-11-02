import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent

def load_artifacts():
    meta_path = ROOT / "pm_meta.joblib"
    model_path = ROOT / "pm_xgb.json"
    if not meta_path.exists() or not model_path.exists():
        raise RuntimeError(
            "Model artifacts missing. Add pm_meta.joblib and pm_xgb.json to the repository root."
        )
    meta = joblib.load(meta_path)
    model = XGBClassifier()
    model.load_model(str(model_path))
    return meta, model

try:
    meta, model = load_artifacts()
    FEATURES = meta["features"]
    TYPES = meta["types"]
    THRESHOLD = float(meta["threshold"])
    TYPE_MODEL = meta["type_model"]
except Exception as exc:
    # Keep the API importable so /health can explain configuration problems.
    meta, model, FEATURES, TYPES, TYPE_MODEL, THRESHOLD = None, None, [], [], None, 0.5
    ARTIFACT_ERROR = str(exc)
else:
    ARTIFACT_ERROR = None

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

class PredictionInput(BaseModel):
    quality_type: str = Field(default="M", pattern="^(L|M|H)$")
    air_temperature: float = Field(ge=250, le=350)
    process_temperature: float = Field(ge=250, le=350)
    rotational_speed: float = Field(gt=0, le=10000)
    torque: float = Field(ge=0, le=200)
    tool_wear: float = Field(ge=0, le=1000)

def build_features(data: PredictionInput) -> pd.DataFrame:
    type_map = {"L": 0, "M": 1, "H": 2}
    row = {
        "Type": type_map[data.quality_type],
        "Air_temperature": data.air_temperature,
        "Process_temperature": data.process_temperature,
        "Rotational_speed": data.rotational_speed,
        "Torque": data.torque,
        "Tool_wear": data.tool_wear,
        "Temp_diff": data.process_temperature - data.air_temperature,
        "Power_W": data.torque * data.rotational_speed * 2 * np.pi / 60,
        "Strain": data.tool_wear * data.torque,
    }
    return pd.DataFrame([row])[FEATURES]

@app.get("/")
def root():
    return {"message": "Predictive Maintenance API", "docs": "/api/docs"}

@app.get("/health")
def health():
    return {"status": "ok" if ARTIFACT_ERROR is None else "artifact_error",
            "artifacts_loaded": ARTIFACT_ERROR is None,
            "detail": ARTIFACT_ERROR}

@app.post("/predict")
def predict(data: PredictionInput) -> dict[str, Any]:
    if ARTIFACT_ERROR:
        raise HTTPException(status_code=503, detail=ARTIFACT_ERROR)
    X = build_features(data)
    risk = float(model.predict_proba(X)[0, 1])
    risk_type = None
    type_confidence = None
    if risk >= THRESHOLD:
        probs = TYPE_MODEL.predict_proba(X)
        scores = {}
        for name, p in zip(TYPES, probs):
            scores[name] = float(p[0, 1]) if p.shape[1] > 1 else 0.0
        if scores:
            risk_type = max(scores, key=scores.get)
            total = sum(scores.values())
            type_confidence = scores[risk_type] / total if total > 1e-9 else 0.0
    type_names = {
        "TWF": "Tool wear",
        "HDF": "Heat dissipation",
        "PWF": "Power",
        "OSF": "Overstrain",
    }
    return {
        "risk": risk,
        "risk_percent": round(risk * 100, 1),
        "threshold": THRESHOLD,
        "threshold_percent": round(THRESHOLD * 100, 1),
        "status": "alert" if risk >= THRESHOLD else "healthy",
        "likely_cause": type_names.get(risk_type, risk_type),
        "diagnosis_confidence_percent": round((type_confidence or 0) * 100, 1),
        "inputs": data.model_dump(),
    }
