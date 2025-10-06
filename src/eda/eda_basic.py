from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.settings import load_config
from src.data.loaders import load_telco_csv


def main() -> None:
	cfg = load_config()
	reports_dir = Path("reports")
	reports_dir.mkdir(parents=True, exist_ok=True)

	df = load_telco_csv(cfg.get("data", "path"))

	# Basic summary
	df.describe(include="all").to_csv(reports_dir / "describe.csv")

	# Churn distribution
	plt.figure(figsize=(5, 4))
	sns.countplot(data=df, x="Churn")
	plt.title("Churn Distribution")
	plt.tight_layout()
	plt.savefig(reports_dir / "churn_distribution.png")
	plt.close()

	# Tenure vs Churn
	plt.figure(figsize=(6, 4))
	sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, fill=True)
	plt.title("Tenure by Churn")
	plt.tight_layout()
	plt.savefig(reports_dir / "tenure_by_churn.png")
	plt.close()

	# MonthlyCharges vs Churn
	plt.figure(figsize=(6, 4))
	sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True)
	plt.title("Monthly Charges by Churn")
	plt.tight_layout()
	plt.savefig(reports_dir / "monthlycharges_by_churn.png")
	plt.close()

	print("EDA artifacts saved in reports/")


if __name__ == "__main__":
	main()


