from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
	customerID: str = Field(...)
	gender: str
	SeniorCitizen: int
	Partner: str
	Dependents: str
	PhoneService: str
	MultipleLines: str
	InternetService: str
	OnlineSecurity: str
	OnlineBackup: str
	DeviceProtection: str
	TechSupport: str
	StreamingTV: str
	StreamingMovies: str
	Contract: str
	PaperlessBilling: str
	PaymentMethod: str
	tenure: int
	MonthlyCharges: float
	TotalCharges: float


class ChurnResponse(BaseModel):
	customerID: str
	churn_probability: float
	churn_pred: int
	model: Optional[str] = None


