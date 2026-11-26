"""In-process FastAPI TestClient compatibility (with and without lifespan)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_bare_testclient_single_requests():
    client = TestClient(app)  # no `with` -> lifespan may not run
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

    r = client.post("/score", json={"values": [0.1, -0.2, 3.0], "timeout_ms": 5000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["score"], float)
    assert body["status"] == "ok"

    r = client.post("/score", json={"values": [1, 2, 3], "timeout_ms": 1})
    assert r.status_code == 408

    r = client.get("/stats")
    assert r.status_code == 200


def test_context_testclient():
    with TestClient(app) as client:
        r = client.post("/score", json={"values": [1, 2, 3, 4], "timeout_ms": 5000})
        assert r.status_code == 200, r.text
        assert isinstance(r.json()["score"], float)
        assert client.get("/health").status_code == 200
