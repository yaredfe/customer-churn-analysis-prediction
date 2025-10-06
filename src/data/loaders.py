from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import ID_COLUMN


def load_telco_csv(path: str | Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	# Some datasets have spaces in TotalCharges, coerce to numeric
	if "TotalCharges" in df.columns:
		df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")
	# Drop rows with missing ID
	df = df.dropna(subset=[ID_COLUMN])
	return df


