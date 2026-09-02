"""Every response tells the browser what this app is allowed to do.

Before this there were no security headers at all -- not a weak policy, none --
on an application that collects blood pressure, mental-health days and free-text
descriptions of symptoms.

The Content-Security-Policy is the one worth testing hardest. /start echoes back
a sentence the visitor typed, and while tests/test_frontend.py checks that it is
escaped, a policy that refuses to execute injected script is the second line
that makes the first one's failure survivable.
"""

import re

import pytest

from app.security import PERMISSIONS, _csp

PAGES = ["/", "/start", "/privacy", "/summary", "/heart_disease/", "/sleep/",
         "/health-score/", "/health/", "/nutrition/", "/healthz"]

REQUIRED = {
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
    "X-Frame-Options",
}


@pytest.mark.parametrize("path", PAGES)
def test_every_page_carries_the_headers(client, path):
    response = client.get(path)
    missing = REQUIRED - set(response.headers.keys())
    assert not missing, f"{path} is missing {sorted(missing)}"


def test_the_error_pages_carry_them_too():
    """A 404 and a 500 are still responses, and an error page that dropped the
    policy would be the most useful page for an attacker to reach."""
    from app.app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    response = app.test_client().get("/no-such-page")
    assert response.status_code == 404
    assert REQUIRED <= set(response.headers.keys())


def test_scripts_may_not_be_inline_without_the_nonce():
    """The policy's whole job. 'unsafe-inline' here would make it decoration."""
    policy = _csp("abc123")
    script_src = next(d for d in policy.split("; ") if d.startswith("script-src"))
    assert "'unsafe-inline'" not in script_src
    assert "'nonce-abc123'" in script_src
    assert "'self'" in script_src


def test_the_nonce_changes_every_response(client):
    """A nonce reused across responses is a nonce an attacker can just read off
    an earlier page and reuse."""
    nonces = set()
    for _ in range(4):
        policy = client.get("/").headers["Content-Security-Policy"]
        nonces.add(re.search(r"'nonce-([^']+)'", policy).group(1))
    assert len(nonces) == 4


def test_the_page_uses_the_nonce_it_was_given(client):
    """The header and the markup have to agree, or the inline scripts that run
    the navigation and the scroll button silently stop working."""
    response = client.get("/")
    body = response.get_data(as_text=True)
    nonce = re.search(
        r"'nonce-([^']+)'", response.headers["Content-Security-Policy"]
    ).group(1)

    inline = re.findall(r"<script(?![^>]*\ssrc=)([^>]*)>", body)
    assert inline, "no inline scripts on the homepage; this test has gone stale"
    for attributes in inline:
        assert f'nonce="{nonce}"' in attributes, (
            f"an inline script carries no nonce, so the CSP will block it: "
            f"<script{attributes}>"
        )


def test_nothing_may_be_loaded_from_another_origin():
    """default-src 'self' with no host allowlist is what keeps the CDNs out for
    good, rather than relying on nobody adding one back."""
    policy = _csp("n")
    assert "default-src 'self'" in policy
    assert "://" not in policy, f"the policy names an external host: {policy}"


def test_an_assessment_url_never_leaves_in_a_referer(client):
    """The paths alone say what somebody was worried about."""
    assert client.get("/heart_disease/").headers["Referrer-Policy"] == "no-referrer"


def test_the_app_gives_up_hardware_it_does_not_use():
    for feature in ("camera", "microphone", "geolocation"):
        assert f"{feature}=()" in PERMISSIONS


def test_hsts_is_sent_over_https_and_not_over_http(client):
    """Sending it over plain HTTP achieves nothing; sending it from a local dev
    server pins the developer's own localhost to HTTPS for a year."""
    assert "Strict-Transport-Security" not in client.get("/").headers

    secure = client.get("/", base_url="https://localhost")
    assert "max-age=31536000" in secure.headers["Strict-Transport-Security"]


def test_the_policy_still_permits_what_the_pages_actually_need():
    """A policy that breaks the app gets removed rather than fixed."""
    policy = _csp("n")
    # base.html's favicon is an inline SVG data: URI.
    assert "img-src 'self' data:" in policy
    # Five pages carry their own <style> block, and ~30 elements a style
    # attribute. See app/security.py for why that is not closed yet.
    assert "style-src 'self' 'unsafe-inline'" in policy
