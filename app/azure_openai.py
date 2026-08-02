"""Azure OpenAI client.

Deliberately a thin wrapper over the REST API using ``requests`` -- already a
dependency -- rather than the SDK. Two reasons, both about the promise on
/privacy:

* **Every byte that leaves this server is assembled here, in one function.** A
  test can assert the exact request body. With an SDK the payload is built
  somewhere in a dependency tree, and "we never send health answers" becomes a
  claim about code nobody in this repo reads.
* No new dependency for a feature that is optional and off by default.

**What may be sent.** Only the free-text sentence a person types into the
"what's bothering you" box on /start, to decide which assessment to show them.
Nothing from any assessment form goes to Azure -- not the answers, not the
results, not the scores. The wording that used to require a live call for
result explanations and doctor questions is generated at build time instead
(ml_model/generate_phrasings.py) and committed, so those pages need no network
at all.

**What happens when it is not configured**, which is the default: nothing.
Every call site falls back to the deterministic path it had before, so the app
behaves exactly as it does today with no Azure credentials present.
"""

from __future__ import annotations

import json
import logging
import os

import requests

log = logging.getLogger(__name__)

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Short on purpose. This sits in front of a page load; if Azure is slow the
# keyword matcher answers instead, and the user never knows the difference.
TIMEOUT = float(os.getenv("AZURE_OPENAI_TIMEOUT", "6"))


class AzureUnavailable(RuntimeError):
    """The call could not be made or could not be trusted."""


def is_configured():
    return bool(ENDPOINT and API_KEY and DEPLOYMENT)


def describe_configuration():
    """For /healthz and the privacy page -- never includes the key."""
    return {
        "configured": is_configured(),
        "endpoint": ENDPOINT or None,
        "deployment": DEPLOYMENT or None,
        "api_version": API_VERSION,
    }


def build_request(messages, *, max_tokens=200, temperature=0.0, json_object=False):
    """Assemble the exact URL, headers and body that will be sent.

    Separated from the call so tests can inspect it without a network. This is
    the single place any outbound payload is constructed.
    """
    url = (f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions"
           f"?api-version={API_VERSION}")
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    return url, headers, body


def complete(messages, *, max_tokens=200, temperature=0.0, json_object=False):
    """One chat completion. Raises ``AzureUnavailable`` rather than propagating.

    Callers are expected to have a working answer without this.
    """
    if not is_configured():
        raise AzureUnavailable("Azure OpenAI is not configured")

    url, headers, body = build_request(
        messages, max_tokens=max_tokens, temperature=temperature,
        json_object=json_object,
    )

    try:
        response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise AzureUnavailable(f"Azure OpenAI timed out after {TIMEOUT}s") from exc
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        raise AzureUnavailable(f"Azure OpenAI request failed ({status})") from exc
    except ValueError as exc:
        raise AzureUnavailable("Azure OpenAI returned unreadable JSON") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AzureUnavailable("Azure OpenAI response had no content") from exc

    if content is None:
        # A content filter returning null is not an error to the HTTP layer.
        raise AzureUnavailable("Azure OpenAI returned no content (filtered?)")
    return content


def complete_json(messages, **kwargs):
    """A completion that must parse as a JSON object."""
    raw = complete(messages, json_object=True, **kwargs)
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise AzureUnavailable("Azure OpenAI did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise AzureUnavailable("Azure OpenAI returned JSON that was not an object")
    return parsed
