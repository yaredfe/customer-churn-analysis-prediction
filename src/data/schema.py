from __future__ import annotations

from typing import List

# Columns based on Telco dataset
ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"
POSITIVE_LABEL: str = "Yes"
NEGATIVE_LABEL: str = "No"

CATEGORICAL_FEATURES: List[str] = [
	"gender",
	"SeniorCitizen",
	"Partner",
	"Dependents",
	"PhoneService",
	"MultipleLines",
	"InternetService",
	"OnlineSecurity",
	"OnlineBackup",
	"DeviceProtection",
	"TechSupport",
	"StreamingTV",
	"StreamingMovies",
	"Contract",
	"PaperlessBilling",
	"PaymentMethod",
]

NUMERIC_FEATURES: List[str] = [
	"tenure",
	"MonthlyCharges",
	"TotalCharges",
]

ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


