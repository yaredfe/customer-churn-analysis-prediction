from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def save_joblib(obj: Any, path: str | Path) -> None:
	path = Path(path)
	ensure_dir(path.parent)
	joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
	return joblib.load(path)


