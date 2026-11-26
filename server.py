"""Compatibility shim for `uvicorn server:app`."""
from app.main import app, create_app  # noqa: F401
