"""Thin launcher for the web UI.

Usage: uv run python -m scripts.serve_ui
"""

import uvicorn


def main() -> None:
    uvicorn.run("webapp.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
