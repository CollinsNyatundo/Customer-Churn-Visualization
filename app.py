"""
app.py
-------
Production Dash entry point.

Data is loaded inside create_app() — NOT at module import time.
This prevents pipeline execution during gunicorn worker startup,
test collection, or any import that doesn't intend to run the server.

Run:
    python app.py                              # development
    gunicorn "app:create_server()" -b 0.0.0.0:8050 -w 2
    make run-dashboard
"""

from __future__ import annotations

import logging
import os

import dash

from src.config import settings
from src.logging_config import configure_logging

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json=os.getenv("LOG_FORMAT", "text") == "json",
)
log = logging.getLogger(__name__)


def create_server() -> dash.Dash:
    """
    Build the Dash app and load all data.
    Called explicitly — never at module import time.
    """
    from src.data.pipeline import MultiSourcePipeline
    from src.features.engineering import build_feature_set
    from src.visualizations.dashboard import create_layout, register_callbacks

    settings.ensure_dirs()

    log.info("Starting data pipeline ...")
    pipeline = MultiSourcePipeline()
    df, meta = pipeline.run()
    df = build_feature_set(df)
    log.info(
        "Pipeline complete: %d rows, %d cols, churn_rate=%.2f%%",
        len(df),
        len(df.columns),
        df["churn"].mean() * 100,
    )

    app = dash.Dash(
        __name__,
        title="Customer Churn Dashboard",
        assets_folder="assets",
    )
    app.layout = create_layout(df)
    register_callbacks(app)
    return app


# ── Gunicorn entry point ─────────────────────────────────────────────────────
# gunicorn "app:server" resolves this at worker start, not at import.
# We use a lazy callable so gunicorn doesn't run the pipeline during
# module load of other workers.

_app: dash.Dash | None = None


def _get_app() -> dash.Dash:
    global _app
    if _app is None:
        _app = create_server()
    return _app


class _LazyServer:
    """WSGI proxy that builds the app on first request, not at import."""

    def __call__(self, environ, start_response):
        return _get_app().server(environ, start_response)


server = _LazyServer()


# ── Development entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_server()
    app.run(
        debug=settings.dash_debug,
        host=settings.dash_host,
        port=settings.dash_port,
    )
