from __future__ import annotations

import os

os.environ.setdefault("API_URL", os.environ.get("API_URL", "http://localhost/api"))

from src.frontend_flask.app import app  # noqa: E402


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)


