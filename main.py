"""Root ASGI entrypoint so `uvicorn main:app` works from the repository root."""
from app.main import app, create_app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    from app.config import get_settings

    cfg = get_settings()
    uvicorn.run("main:app", host=cfg.host, port=cfg.port, log_level="info")
