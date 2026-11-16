"""Entry point: ``python -m backend.main`` or ``uvicorn backend.api:app``."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
