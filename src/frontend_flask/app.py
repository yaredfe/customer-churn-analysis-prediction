from __future__ import annotations

import os
from typing import Any, Dict

import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def get_schema() -> Dict[str, Any]:
	resp = requests.get(f"{API_URL}/schema", timeout=5)
	resp.raise_for_status()
	return resp.json()


@app.get("/")
def index():
    schema = get_schema()
    return render_template("form.html", schema=schema)


@app.post("/predict")
def predict():
	# Build payload reading allowed fields from schema to avoid drift
	schema = get_schema()
	payload: Dict[str, Any] = {"customerID": request.form.get("customerID", "web-user")}
	for feat in schema.get("numeric_features", []):
		val = request.form.get(feat)
		payload[feat] = float(val) if val is not None else None
	for feat, cats in schema.get("categorical_features", {}).items():
		payload[feat] = request.form.get(feat)

	resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
	data = resp.json()
	return render_template("result.html", result=data)


    


@app.get("/health")
def health():
	return jsonify({"status": "ok"})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
