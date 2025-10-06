from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, POSITIVE_LABEL


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	X = df.drop(columns=[TARGET_COLUMN])
	y = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
	return X, y


def build_preprocessor() -> ColumnTransformer:
	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
		]
	)
	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, NUMERIC_FEATURES),
			("cat", categorical_pipeline, CATEGORICAL_FEATURES),
		],
		remainder="drop",
	)
	return preprocessor


