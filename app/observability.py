"""Making a production error traceable back to the person who hit it.

The app logs properly, but until now nothing tied a log line to a request. When
someone says "it broke", there was no way to find which of a day's requests
they meant.

Every request gets a short id. It goes on every log line emitted while handling
that request, and onto the error page the user sees, so "it broke, the code was
7f3a2b1c" is enough to find the traceback.

Sentry is optional and is only wired up when SENTRY_DSN is set *and* the SDK is
installed. It is not in requirements.txt: a health app that stores nothing
should not acquire a hard dependency on a third party that would receive its
stack traces. Anyone who wants it can `pip install sentry-sdk` and set the DSN.

Nothing here records what anyone entered. Request ids are random, tied to
nothing, and discarded when the response is sent.
"""

from __future__ import annotations

import logging
import secrets
import time

from flask import g, request

log = logging.getLogger(__name__)


def install_record_factory():
    """Give every log record a ``request_id``, whoever creates it.

    Done at the record factory rather than as a handler filter. A filter only
    covers the handlers attached when it was installed, so a handler added
    later -- by gunicorn, by a library, by a test -- would produce records
    without the attribute, and the format string referencing it would fail at
    the moment something is trying to report an error. Setting it here means
    the attribute always exists.

    Safe to call more than once.
    """
    existing = logging.getLogRecordFactory()
    if getattr(existing, "_lifepulse_wrapped", False):
        return

    def factory(*args, **kwargs):
        record = existing(*args, **kwargs)
        try:
            record.request_id = g.get("request_id", "-")
        except RuntimeError:  # outside an application context
            record.request_id = "-"
        return record

    factory._lifepulse_wrapped = True
    logging.setLogRecordFactory(factory)


# Installed on import so records created before create_app() still carry the
# attribute -- otherwise the very first startup warning would fail to format.
install_record_factory()


def _install_sentry(app):
    dsn = app.config.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
    except ImportError:
        log.warning(
            "SENTRY_DSN is set but sentry-sdk is not installed; "
            "run `pip install sentry-sdk` or unset the variable."
        )
        return False

    sentry_sdk.init(
        dsn=dsn,
        integrations=[FlaskIntegration()],
        environment=app.config.get("ENVIRONMENT", "production"),
        # Health answers are submitted in request bodies. Never send them.
        send_default_pii=False,
        max_request_body_size="never",
        # Nor in stack frames. This one is easy to miss: blocking the request
        # body is not enough, because Sentry attaches every frame's local
        # variables to an exception by default. A 500 anywhere in an assessment
        # route has the form dict sitting in `collect()`'s locals and the
        # feature vector in `ModelBundle._matrix()`'s -- so the answers would
        # have reached Sentry through the traceback with the body still
        # correctly withheld.
        include_local_variables=False,
        traces_sample_rate=0.0,
    )
    log.info("Sentry enabled (request bodies and frame locals are never sent)")
    return True


def init_app(app):
    """Wire request ids, timing, and optional Sentry into ``app``."""
    install_record_factory()
    sentry_on = _install_sentry(app)

    @app.before_request
    def _start_request():
        g.request_id = secrets.token_hex(4)
        g.request_started = time.perf_counter()

    @app.after_request
    def _log_request(response):
        started = g.get("request_started")
        elapsed_ms = (time.perf_counter() - started) * 1000 if started else 0

        # Only the method, path and outcome. Never the form body, which is
        # where every health answer lives.
        level = logging.WARNING if response.status_code >= 500 else logging.INFO
        log.log(
            level, "%s %s -> %s in %.0fms",
            request.method, request.path, response.status_code, elapsed_ms,
        )
        response.headers["X-Request-Id"] = g.get("request_id", "-")
        return response

    app.config["SENTRY_ENABLED"] = sentry_on
    return app


def current_request_id():
    """The id for this request, or "-" outside one."""
    try:
        return g.get("request_id", "-")
    except RuntimeError:
        return "-"
