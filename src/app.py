"""Local web app. Start with: python src/app.py (no external model API)."""

import argparse
import logging
import os
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from phase3_pipeline import RAW_CORPUS
from product import InputError, load_verifier, validate_input

LOGGER = logging.getLogger(__name__)


class ModelState:
    """One background startup job and one bounded inference at a time."""

    def __init__(self, loader):
        self.loader = loader
        self.lock = Lock()
        self.inference_lock = Lock()
        self.state = "not_started"
        self.error = None
        self.verifier = None

    def start(self):
        with self.lock:
            if self.state not in ("not_started", "error"):
                return
            self.state, self.error = "loading", None
        Thread(target=self._load, daemon=True, name="citationguard-model").start()

    def _load(self):
        try:
            verifier = self.loader()
        except Exception as error:
            LOGGER.exception("Model setup failed")
            message = str(error) if isinstance(error, RuntimeError) else (
                "Model setup failed. Check the local server log, installed requirements and data/processed training files, then Retry setup."
            )
            with self.lock:
                self.state, self.error = "error", message
        else:
            with self.lock:
                self.verifier, self.state = verifier, "ready"

    def snapshot(self):
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "model": self.verifier.metadata if self.state == "ready" else None,
            }


def create_app(loader=None, start_model=True):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    data_dir = Path(os.environ.get("CITATIONGUARD_DATA_DIR", str(RAW_CORPUS.parent)))
    state = ModelState(loader or (lambda: load_verifier(data_dir)))
    app.extensions["model_state"] = state

    @app.after_request
    def secure_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(state.snapshot())

    @app.post("/api/retry")
    def retry():
        state.start()
        return jsonify(state.snapshot()), 202

    @app.post("/api/verify")
    def verify():
        try:
            claim, doc_id = validate_input(request.get_json())
        except (BadRequest, UnsupportedMediaType):
            return jsonify(error="Send valid JSON with Content-Type: application/json."), 400
        except InputError as error:
            return jsonify(error=str(error)), 400
        if state.snapshot()["state"] != "ready":
            return jsonify(error="The model is not ready. Check setup status and retry."), 503
        if not state.inference_lock.acquire(blocking=False):
            return jsonify(error="Another claim is being checked. Please retry shortly."), 503
        try:
            return jsonify(state.verifier.verify(claim, doc_id))
        except InputError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            LOGGER.exception("Verification failed")
            return jsonify(error="Verification failed; no verdict was produced. Check the local server log and retry."), 500
        finally:
            state.inference_lock.release()

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="Request too large. Send one claim of at most 2000 characters."), 413

    if start_model:
        state.start()
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Loopback only. This development server must not be exposed to the internet.
    create_app().run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
