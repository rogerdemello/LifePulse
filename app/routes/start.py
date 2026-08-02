"""Symptom-led entry point.

People arrive with something they've noticed, not with a category in mind.
This asks what's bothering them and points at the right assessment — and
stops entirely if what they describe needs emergency care instead.

When Azure OpenAI is configured the routing is conversational: the agent may
ask up to two clarifying questions before committing. Three things are true of
every turn, not just the first:

* **The emergency check runs first, locally, on everything the person has
  typed so far.** Someone who types "just tired" and then answers a follow-up
  with "and crushing chest pain" gets the stop sign on that second turn. It is
  keyword-based and cannot be overridden by the model, and it does not depend
  on Azure being reachable.
* **Nothing is stored.** The exchange rides in hidden fields on the form and
  comes back with the next post, the same way the red-flag interstitial
  replays a pending submission. There is no session and no database row.
* **Any failure falls back to keyword matching** over everything said so far,
  so an unconfigured, slow or wrong Azure is an ordinary condition.
"""

from flask import Blueprint, render_template, request

from app.ml.triage import CONCERNS, check_emergency, converse, read_turns
from app.ratelimit import rate_limit

start_bp = Blueprint("start", __name__)


@start_bp.route("/start", methods=["GET", "POST"])
@rate_limit(limit=40, window=60)
def start():
    if request.method != "POST":
        return render_template("start.html", concerns=CONCERNS, turns=[])

    turns = read_turns(request.form.getlist("turn"))
    described = (request.form.get("concern") or "").strip()
    if described:
        turns.append({"role": "user", "text": described[:500]})

    said = " ".join(t["text"] for t in turns if t["role"] == "user").strip()

    # Before anything else, and before any network call. Deterministic keyword
    # rules on everything typed so far -- including answers to the agent's own
    # follow-up questions, which is the turn this most needs to cover.
    emergencies = check_emergency(said)
    if emergencies:
        return render_template(
            "start.html",
            concerns=CONCERNS,
            turns=turns,
            described=said,
            emergencies=emergencies,
        ), 200

    if not said:
        return render_template(
            "start.html", concerns=CONCERNS, turns=[],
            error="Tell us what's bothering you, or pick from the list below.",
        ), 400

    outcome = converse(turns)

    if outcome.action == "ask":
        turns.append({"role": "agent", "text": outcome.question})
        return render_template(
            "start.html",
            concerns=CONCERNS,
            turns=turns,
            described=said,
            asking=outcome.question,
        )

    return render_template(
        "start.html",
        concerns=CONCERNS,
        turns=turns,
        described=said,
        matches=outcome.matches,
        method=outcome.method,
    )
