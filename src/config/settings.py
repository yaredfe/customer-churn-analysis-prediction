from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
	def __init__(self, data: Dict[str, Any]) -> None:
		self._data = data

	def get(self, *path: str, default: Any | None = None) -> Any:
		node: Any = self._data
		for key in path:
			if not isinstance(node, dict) or key not in node:
				return default
			node = node[key]
		return node

	@property
	def data(self) -> Dict[str, Any]:
		return self._data


def load_config(config_path: str | Path = Path("src/config/config.yaml")) -> Config:
	with open(config_path, "r", encoding="utf-8") as f:
		raw = yaml.safe_load(f)
	return Config(raw)


