"""A small in-process rate limiter for the model endpoints.

Every prediction runs a real model on the server's CPU, and the endpoints are
unauthenticated by design. On a small instance that is a cheap way for one
client to make the site unusable for everyone else -- not through malice
necessarily, a runaway script does it just as well.

Deliberately dependency-free and in-memory:

* State is per worker process. With N gunicorn workers the effective ceiling is
  N x the configured limit. That is fine for the purpose -- keeping one client
  from saturating the box -- and wrong for anything that needs an exact quota.
* State is lost on restart, so a deploy resets every counter.

If this ever needs to be exact or shared, it wants Redis and Flask-Limiter.
Until then this costs nothing and removes the obvious failure mode.

Only IP addresses and timestamps are held, never anything submitted. Entries
expire on their own, so nothing accumulates.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from functools import wraps

from flask import current_app, render_template, request

log = logging.getLogger(__name__)

_hits = defaultdict(deque)

# Generous: a person filling in forms will not come close, while a script
# hammering the endpoint is stopped quickly.
DEFAULT_LIMIT = 30
DEFAULT_WINDOW = 60  # seconds

# Stop the dict growing without bound on a long-lived process.
_MAX_TRACKED_CLIENTS = 10_000


def _client_id():
    """Best-effort client identity.

    Render and similar platforms sit behind a proxy, so the direct peer address
    is the load balancer. Prefer the left-most X-Forwarded-For entry, which is
    the original client. It is spoofable -- this is throttling, not
    authentication, and a spoofed value only splits an attacker's own bucket.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _evict_stale(now, window):
    """Drop clients with no recent activity once the table gets large."""
    for key in [k for k, hits in _hits.items() if not hits or now - hits[-1] > window]:
        del _hits[key]


def is_allowed(client, limit=DEFAULT_LIMIT, window=DEFAULT_WINDOW):
    """Record a hit for ``client`` and report whether it stays within the limit."""
    now = time.monotonic()
    hits = _hits[client]

    while hits and now - hits[0] > window:
        hits.popleft()

    if len(_hits) > _MAX_TRACKED_CLIENTS:
        _evict_stale(now, window)

    if len(hits) >= limit:
        return False

    hits.append(now)
    return True


def rate_limit(limit=DEFAULT_LIMIT, window=DEFAULT_WINDOW):
    """Throttle a view by client address.

    Applies to POST only: browsing the forms is free, running the models is
    what costs something.
    """

    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if request.method != "POST":
                return view(*args, **kwargs)

            if current_app.config.get("RATELIMIT_DISABLED"):
                return view(*args, **kwargs)

            client = _client_id()
            if not is_allowed(client, limit, window):
                log.warning("rate limit hit by %s on %s", client, request.path)
                response = render_template(
                    "error.html",
                    title="Slow down a moment",
                    message=(
                        "You've submitted a lot of assessments in a short time, so "
                        "we've paused for a minute to keep the service responsive "
                        "for everyone. Please try again shortly — nothing you "
                        "entered was saved."
                    ),
                )
                return response, 429, {"Retry-After": str(window)}

            return view(*args, **kwargs)

        return wrapper

    return decorator


def reset():
    """Clear all counters. Used by tests."""
    _hits.clear()
