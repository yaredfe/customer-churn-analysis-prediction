from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
	from xgboost import XGBClassifier  # type: ignore
	_xgb_available = True
except Exception:  # pragma: no cover
	_xgb_available = False
	XGBClassifier = None  # type: ignore

try:
	from lightgbm import LGBMClassifier  # type: ignore
	_lgbm_available = True
except Exception:  # pragma: no cover
	_lgbm_available = False
	LGBMClassifier = None  # type: ignore


def get_estimator(name: str):
	name = name.lower()
	if name == "logistic_regression":
		return LogisticRegression
	if name == "random_forest":
		return RandomForestClassifier
	if name == "xgboost" and _xgb_available and XGBClassifier is not None:
		return XGBClassifier
	if name == "lightgbm" and _lgbm_available and LGBMClassifier is not None:
		return LGBMClassifier
	raise ValueError(f"Unknown or unavailable estimator: {name}")


