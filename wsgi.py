"""WSGI entry point for production: `gunicorn wsgi:app`."""

import logging

from app.app import create_app

# Route application logs to gunicorn's handlers when running under it, so
# warnings from the feature-contract checks actually reach the platform log.
_gunicorn = logging.getLogger("gunicorn.error")
if _gunicorn.handlers:
    logging.getLogger().handlers = _gunicorn.handlers
    logging.getLogger().setLevel(_gunicorn.level)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

app = create_app()
