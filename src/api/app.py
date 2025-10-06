from __future__ import annotations

from pathlib import Path
from typing import List

import json
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import ChurnRequest, ChurnResponse
from src.config.settings import load_config
from src.utils.io import load_joblib

app = FastAPI(title="Customer Churn Prediction API")

_cfg = load_config()
_model_dir = Path(_cfg.get("project", "artifacts_dir")) / "models"
_model_path = _model_dir / "best_model.joblib"
_schema_path = _model_dir / "schema.json"
_model = None
_model_name = None
_schema_cache = None


def _load_model():
	global _model, _model_name
	if _model is None:
		_model = load_joblib(_model_path)
		_model_name = _model.named_steps["model"].__class__.__name__


def _load_schema():
	global _schema_cache
	if _schema_cache is None:
		if not _schema_path.exists():
			raise FileNotFoundError("schema.json not found. Train the model first.")
		with open(_schema_path, "r", encoding="utf-8") as f:
			_schema_cache = json.load(f)
	return _schema_cache


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/schema")
def schema() -> dict:
	try:
		return _load_schema()
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
	_load_model()
	# Convert to DataFrame expected by pipeline
	row = payload.model_dump()
	df = pd.DataFrame([row]).drop(columns=["Churn"], errors="ignore")
	proba = float(_model.predict_proba(df)[:, 1][0])
	pred = int(proba >= 0.5)
	return ChurnResponse(customerID=payload.customerID, churn_probability=proba, churn_pred=pred, model=_model_name)
