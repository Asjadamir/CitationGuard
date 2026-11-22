"""Execute HTTP routes and startup state, not implementation source checks."""

from threading import Event
import time

import pytest

from app import create_app


def wait_for_state(state, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if state.snapshot()["state"] == expected:
            return
        time.sleep(.01)
    pytest.fail(f"Setup did not become {expected}: {state.snapshot()}")


@pytest.fixture
def app(tiny_verifier):
    app = create_app(loader=lambda: tiny_verifier, start_model=False)
    app.extensions["model_state"].start()
    wait_for_state(app.extensions["model_state"], "ready")
    return app


def test_home_and_static_assets(app):
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert home.mimetype == "text/html"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert home.headers["Cache-Control"] == "no-store"
    assert home.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in home.headers["Content-Security-Policy"]


def test_http_claim_to_evidence_and_invalid_document(app):
    client = app.test_client()
    assert client.get("/api/status").json["state"] == "ready"
    response = client.post("/api/verify", json={"claim": "Copper reduces fever.", "doc_id": "17"})
    assert response.status_code == 200
    assert response.json["label"] == "SUPPORT"
    assert response.json["evidence"][0]["text"] == "Copper reduces fever."
    response = client.post("/api/verify", json={"claim": "Copper reduces fever.", "doc_id": "999"})
    assert response.status_code == 400
    assert "error" in response.json
    assert "label" not in response.json


@pytest.mark.parametrize("request_kwargs,expected_status", [
    ({"json": {"claim": ""}}, 400),
    ({"json": []}, 400),
    ({"data": "not json", "content_type": "application/json"}, 400),
    ({"data": "plain text"}, 400),
    ({"json": {"claim": "x" * 2001}}, 400),
    ({"json": {"claim": "x" * 17000}}, 413),
])
def test_bad_requests_return_errors_without_verdicts(app, request_kwargs, expected_status):
    response = app.test_client().post("/api/verify", **request_kwargs)
    assert response.status_code == expected_status
    assert "error" in response.json
    assert "label" not in response.json


def test_unready_returns_503_without_inference(tiny_verifier):
    app = create_app(loader=lambda: tiny_verifier, start_model=False)
    response = app.test_client().post("/api/verify", json={"claim": "A test claim."})
    assert response.status_code == 503
    assert "label" not in response.json
    assert tiny_verifier.model.frames == []


def test_background_loading_error_and_recovery(tiny_verifier):
    release = Event()
    calls = []

    def loader():
        calls.append(True)
        if len(calls) == 1:
            release.wait(timeout=3)
            raise RuntimeError("SciFact corpus is missing. Run data setup.")
        return tiny_verifier

    app = create_app(loader=loader)
    state = app.extensions["model_state"]
    client = app.test_client()
    assert client.get("/api/status").json["state"] == "loading"
    client.post("/api/retry")  # Does not create a second concurrent loader.
    assert len(calls) == 1
    release.set()
    wait_for_state(state, "error")
    assert "missing" in client.get("/api/status").json["error"]
    assert client.post("/api/retry").status_code == 202
    wait_for_state(state, "ready")
    assert len(calls) == 2


def test_busy_and_inference_failure_release_capacity(app, tiny_verifier):
    state = app.extensions["model_state"]
    client = app.test_client()
    state.inference_lock.acquire()
    try:
        busy = client.post("/api/verify", json={"claim": "Copper reduces fever."})
        assert busy.status_code == 503
        assert "label" not in busy.json
    finally:
        state.inference_lock.release()
    original = tiny_verifier.model.predict_proba

    def fail(_frame):
        raise ValueError("private internal diagnostic")

    tiny_verifier.model.predict_proba = fail
    failed = client.post("/api/verify", json={"claim": "Copper reduces fever."})
    assert failed.status_code == 500
    assert failed.json == {"error": "Verification failed; no verdict was produced. Check the local server log and retry."}
    tiny_verifier.model.predict_proba = original
    assert client.post("/api/verify", json={"claim": "Copper reduces fever."}).status_code == 200
