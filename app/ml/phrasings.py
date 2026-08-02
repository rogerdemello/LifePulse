"""Result copy written ahead of time, looked up locally.

``phrasings.json`` is produced by ml_model/generate_phrasings.py on a
developer's machine and committed. At runtime this reads it and nothing else --
no network, no per-user call, and therefore no assessment answers leaving the
server. The person's own numbers are substituted here, locally.

The file is optional. When it is absent, or when it has no entry for a
particular field, every call site keeps the wording it already had. That is
deliberate: the app must not depend on generated content existing, or a missing
file becomes a broken page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PATH = Path(__file__).resolve().parent / "phrasings.json"

_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = json.loads(PATH.read_text("utf-8"))
    except FileNotFoundError:
        _cache = {}
    except ValueError:
        log.warning("phrasings.json is not valid JSON; falling back to built-in copy")
        _cache = {}
    return _cache


def available():
    data = _load()
    return bool(data.get("explanation") or data.get("questions"))


def explanation_for(field, direction):
    """A pre-written sentence for this factor, or ``None`` to use the default."""
    return _load().get("explanation", {}).get(f"{field}|{direction}")


def questions_for_band(topic, band):
    """Pre-written questions for a result band, or ``[]`` to use the defaults."""
    return list(_load().get("questions", {}).get(f"{topic}|{band}", []))


def reset():
    """Drop the cache. Used by tests."""
    global _cache
    _cache = None
