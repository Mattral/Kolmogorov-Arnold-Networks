"""Supervisor-managed entrypoint.

Re-exports the FastAPI app from ``api.app`` so the platform's supervisor (which
runs ``uvicorn server:app`` from ``/app/backend``) can serve the same REST API
used by the Docker image.
"""
from __future__ import annotations

import os
import sys

# Make the src/ layout importable without an explicit install.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_ROOT, "src"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.app import app  # noqa: E402,F401  (re-exported for uvicorn)
