"""Throttling on the endpoints that run models.

Every prediction is real CPU work behind an unauthenticated endpoint, so one
client -- or one runaway script -- can make the site unusable for everyone.
"""

import pytest

from app.ratelimit import DEFAULT_LIMIT, DEFAULT_WINDOW, is_allowed, reset

FORM = {
    "Age": "32", "Gender": "Female", "SleepHours": "6.5", "WaterIntake": "5",
    "SkippedMeals": "No", "Caffeine": "2", "Stress": "5", "ScreenTime": "8",
    "PhysicalActivity": "0", "Menstruating": "1",
}


@pytest.fixture(autouse=True)
def _clean():
    reset()
    yield
    reset()


def test_a_burst_is_eventually_throttled(throttled_client):
    statuses = [
        throttled_client.post("/migraine/", data=FORM).status_code
        for _ in range(DEFAULT_LIMIT + 5)
    ]
    assert 200 in statuses, "nothing got through at all"
    assert 429 in statuses, "the limiter never engaged"
    # Once throttled it must stay throttled for the rest of the burst.
    assert statuses[-1] == 429


def test_the_throttle_response_is_readable_and_reassuring(throttled_client):
    for _ in range(DEFAULT_LIMIT + 2):
        response = throttled_client.post("/migraine/", data=FORM)
        if response.status_code == 429:
            break

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == str(DEFAULT_WINDOW)
    body = response.get_data(as_text=True)
    assert "Traceback" not in body
    # A health tool should say what happened to the data, not just refuse.
    assert "was saved" in body


def test_browsing_the_forms_is_never_throttled(throttled_client):
    """Only POSTs cost anything; reading a form must always work."""
    for _ in range(DEFAULT_LIMIT * 2):
        assert throttled_client.get("/migraine/").status_code == 200


def test_clients_are_counted_separately():
    for _ in range(DEFAULT_LIMIT):
        assert is_allowed("10.0.0.1")
    assert not is_allowed("10.0.0.1")
    # A different client must be unaffected.
    assert is_allowed("10.0.0.2")


def test_the_window_slides(monkeypatch):
    import app.ratelimit as ratelimit

    now = [1000.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])

    for _ in range(DEFAULT_LIMIT):
        assert ratelimit.is_allowed("10.0.0.9")
    assert not ratelimit.is_allowed("10.0.0.9")

    now[0] += DEFAULT_WINDOW + 1
    assert ratelimit.is_allowed("10.0.0.9"), "the window never expired"


def test_forwarded_header_identifies_the_client_behind_a_proxy(throttled_client):
    """On Render the peer address is the load balancer, not the user.

    Without this every visitor shares one bucket and the first busy minute
    locks out the whole site.
    """
    for _ in range(DEFAULT_LIMIT + 2):
        blocked = throttled_client.post(
            "/migraine/", data=FORM, headers={"X-Forwarded-For": "203.0.113.5"}
        )
        if blocked.status_code == 429:
            break
    assert blocked.status_code == 429

    other = throttled_client.post(
        "/migraine/", data=FORM, headers={"X-Forwarded-For": "203.0.113.99"}
    )
    assert other.status_code == 200, "one client's burst blocked everyone else"
