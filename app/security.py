"""Response headers that tell the browser what this app is allowed to do.

There were none. Not a weak policy -- no ``Content-Security-Policy``, no
``X-Content-Type-Options``, no ``Referrer-Policy``, no ``Permissions-Policy``,
no HSTS -- on an application that collects blood-pressure readings, mental-health
days, and free-text descriptions of symptoms.

Each header here closes something specific:

``Content-Security-Policy``
    The one that matters. Scripts may load only from this origin or carry the
    per-response nonce, so an injected ``<script>`` does not run even if markup
    escaping fails somewhere. That second line of defence is worth having
    precisely because /start echoes back a sentence the visitor typed.

``Referrer-Policy``
    An assessment URL should not travel in a ``Referer`` header to anywhere,
    and ``no-referrer`` is the only value that guarantees it. The paths alone
    say what someone was worried about.

``X-Content-Type-Options``
    Stops a browser deciding for itself that a response is JavaScript.

``Permissions-Policy``
    This app has no use for a camera, a microphone or a location, so it gives
    them up rather than leaving the decision to a future dependency.

``Strict-Transport-Security``
    Sent only over HTTPS. Sending it over plain HTTP does nothing, and sending
    it from a local development server would pin ``localhost`` to HTTPS in the
    developer's browser for a year -- which is a genuinely irritating thing to
    do to somebody, and hard to undo.
"""

from __future__ import annotations

import secrets

from flask import g, request

# One year, and the app is served over HTTPS on Render. `preload` is
# deliberately absent: it is a one-way door enforced by the browser vendors
# rather than by this header, and it should be a deployment decision.
HSTS = "max-age=31536000; includeSubDomains"

PERMISSIONS = ", ".join(
    f"{feature}=()"
    for feature in (
        "accelerometer", "camera", "geolocation", "gyroscope", "magnetometer",
        "microphone", "payment", "usb", "interest-cohort",
    )
)


def _csp(nonce):
    """The policy, assembled per response so the nonce is never reused.

    ``style-src`` still allows inline styles, and that is a real gap rather than
    an oversight. The templates carry about thirty ``style="max-width: 42rem"``
    attributes and five per-page ``<style>`` blocks; a nonce cannot cover style
    *attributes* at all, so closing this means moving every one of them into a
    stylesheet. Worth doing, and not worth doing halfway -- an injected style
    can restyle a page but cannot execute, which is why scripts were the half to
    lock down first.
    """
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' 'nonce-{nonce}'",
        "style-src 'self' 'unsafe-inline'",
        # data: is for the inline SVG favicon in base.html.
        "img-src 'self' data:",
        "font-src 'self'",
        # No XHR anywhere, but if one is added it should stay on this origin.
        "connect-src 'self'",
        # Every form on this site posts to this site.
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "object-src 'none'",
    ])


def init_app(app):
    """Attach the headers, and expose the nonce to templates."""

    @app.before_request
    def _make_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.after_request
    def _set_headers(response):
        nonce = g.get("csp_nonce", "")
        response.headers["Content-Security-Policy"] = _csp(nonce)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = PERMISSIONS
        # Legacy, and harmless: modern browsers use frame-ancestors above.
        response.headers["X-Frame-Options"] = "DENY"
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = HSTS
        return response

    @app.context_processor
    def _expose_nonce():
        return {"csp_nonce": g.get("csp_nonce", "")}

    return app
