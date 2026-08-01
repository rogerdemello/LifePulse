"""Request tracing, and the promise that it records nothing personal.

The app logged properly but nothing tied a line to a request, so "it broke" was
untraceable. Every request now carries a short id that appears on its log lines
and on the error page, which is enough to find the traceback.

The constraint that shapes all of it: this is a health app that stores nothing.
Tracing must not become a back door through which answers get recorded.
"""

import logging

import pytest

from app import observability

FORM = {
    "snoring": "3", "gasping": "1", "sleepiness": "2", "insomnia_nights": "5",
    "insomnia_months": "1", "insomnia_impact": "1", "sleep_hours": "4.5",
}


def test_every_response_carries_a_request_id(client):
    response = client.get("/sleep/")
    assert response.headers.get("X-Request-Id")
    assert response.headers["X-Request-Id"] != "-"


def test_ids_differ_between_requests(client):
    first = client.get("/").headers["X-Request-Id"]
    second = client.get("/").headers["X-Request-Id"]
    assert first != second


def test_every_log_record_has_a_request_id_attribute(caplog):
    """Format strings reference %(request_id)s, so it must always exist.

    Installed at the log-record factory rather than as a handler filter: a
    filter only covers handlers attached when it ran, so one added later --
    by gunicorn, a library, or a test -- would produce records without the
    attribute, and the format would fail exactly when something is trying to
    report an error.
    """
    observability.install_record_factory()
    with caplog.at_level(logging.INFO):
        logging.getLogger("test.outside.request").info("no request context here")
    assert caplog.records
    assert all(hasattr(record, "request_id") for record in caplog.records)


def test_the_record_factory_is_idempotent():
    """create_app may run more than once in a process; wrappers must not stack."""
    observability.install_record_factory()
    first = logging.getLogRecordFactory()
    observability.install_record_factory()
    observability.install_record_factory()
    assert logging.getLogRecordFactory() is first


def test_request_logs_record_the_route_but_never_the_answers(client, caplog):
    """A log that captured form bodies would quietly undo "nothing is stored"."""
    with caplog.at_level(logging.INFO):
        client.post("/sleep/", data=FORM)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "/sleep/" in logged
    for value in ("snoring", "insomnia_nights", "gasping"):
        assert value not in logged, f"{value} leaked into the logs"


def test_a_server_error_shows_a_quotable_reference(client, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.routes.sleep.assess_apnea", explode)
    response = client.post("/sleep/", data=FORM)

    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Reference" in body
    assert response.headers["X-Request-Id"] in body
    # Still no traceback or exception text for the user.
    assert "Traceback" not in body
    assert "boom" not in body


def test_user_errors_get_no_reference(client):
    """There is nothing to report when someone simply left a field blank."""
    body = client.post("/sleep/", data={"snoring": "1"}).get_data(as_text=True)
    assert "Reference" not in body


def test_sentry_stays_off_without_a_dsn(app):
    assert app.config.get("SENTRY_ENABLED") is False


def test_sentry_is_not_a_hard_dependency():
    """It receives stack traces from a health app; that should be opt-in.

    Nothing may import sentry_sdk at module scope, or the app stops booting
    for anyone who has not installed it.
    """
    from pathlib import Path

    source = (Path(observability.__file__)).read_text(encoding="utf-8")
    top_level = [
        line for line in source.splitlines()
        if line.startswith("import sentry") or line.startswith("from sentry")
    ]
    assert not top_level, "sentry_sdk must only be imported inside _install_sentry"

    requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
    assert "sentry" not in requirements.read_text(encoding="utf-8").lower()


def test_sentry_config_never_sends_request_bodies():
    """Health answers live in request bodies."""
    from pathlib import Path

    source = Path(observability.__file__).read_text(encoding="utf-8")
    assert "send_default_pii=False" in source
    assert 'max_request_body_size="never"' in source
