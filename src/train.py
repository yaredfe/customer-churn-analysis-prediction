from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config.settings import load_config
from src.data.loaders import load_telco_csv
from src.pipeline.preprocess import build_preprocessor, split_features_target
from src.models.model_zoo import get_estimator
from src.utils.io import ensure_dir, save_joblib
from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def main() -> None:
	cfg = load_config()
	artifacts_dir = Path(cfg.get("project", "artifacts_dir"))
	ensure_dir(artifacts_dir)

	# Load data
	csv_path = cfg.get("data", "path")
	df = load_telco_csv(csv_path)

	# Keep only necessary columns
	target_col = cfg.get("data", "target")
	assert target_col in df.columns, "Target column missing in dataset"
	X, y = split_features_target(df)

	x_train, x_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=cfg.get("data", "test_size"),
		random_state=cfg.get("data", "random_state"),
		stratify=y,
	)

	preprocessor = build_preprocessor()

	# Build model search space
	models: List[Dict[str, Any]] = cfg.get("training", "models")
	best_model_name = None
	best_auc = -1.0
	best_pipeline: Pipeline | None = None
	metrics_report: Dict[str, Any] = {}

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.get("data", "random_state"))

	for model_cfg in models:
		name = model_cfg["name"]
		estimator_key = model_cfg["estimator"]
		Estimator = get_estimator(estimator_key)
		pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", Estimator())])
		# Prefix param grid with model__
		param_grid = {f"model__{k}": v for k, v in model_cfg.get("params", {}).items()}

		search = GridSearchCV(
			estimator=pipeline,
			param_grid=param_grid,
			scoring="roc_auc",
			cv=cv,
			n_jobs=-1,
			verbose=0,
		)
		search.fit(x_train, y_train)

		# Evaluate on test
		y_prob = search.best_estimator_.predict_proba(x_test)[:, 1]
		y_pred = (y_prob >= 0.5).astype(int)

		auc = roc_auc_score(y_test, y_prob)
		acc = accuracy_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred)
		rec = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)

		metrics_report[name] = {
			"best_params": search.best_params_,
			"roc_auc": auc,
			"accuracy": acc,
			"precision": prec,
			"recall": rec,
			"f1": f1,
		}

		if auc > best_auc:
			best_auc = auc
			best_model_name = name
			best_pipeline = search.best_estimator_

	assert best_pipeline is not None

	# Save best model and metrics
	model_dir = artifacts_dir / "models"
	ensure_dir(model_dir)
	model_path = model_dir / "best_model.joblib"
	save_joblib(best_pipeline, model_path)

	# Extract and save schema metadata for frontend alignment
	pre = best_pipeline.named_steps["preprocessor"]
	cat_pipe = pre.named_transformers_["cat"]
	ohe = cat_pipe.named_steps["onehot"]
	cat_map = {feat: [str(v) for v in cats] for feat, cats in zip(CATEGORICAL_FEATURES, ohe.categories_)}
	schema = {
		"categorical_features": cat_map,
		"numeric_features": NUMERIC_FEATURES,
	}
	with open(model_dir / "schema.json", "w", encoding="utf-8") as sf:
		json.dump(schema, sf, indent=2)

	reports_dir = Path("reports")
	ensure_dir(reports_dir)
	with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
		json.dump({
			"best_model": best_model_name,
			"best_roc_auc": best_auc,
			"all_models": metrics_report,
		}, f, indent=2)

	print(f"Saved best model to {model_path}")
	print(f"Metrics report written to {reports_dir / 'metrics.json'}")
	print(f"Saved schema to {model_dir / 'schema.json'}")


if __name__ == "__main__":
	main()
