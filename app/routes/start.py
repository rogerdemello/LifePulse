"""Symptom-led entry point.

People arrive with something they've noticed, not with a category in mind.
This asks what's bothering them and points at the right assessment — and
stops entirely if what they describe needs emergency care instead.
"""

from flask import Blueprint, render_template, request

from app.ml.triage import CONCERNS, check_emergency, match_concerns
from app.ratelimit import rate_limit

start_bp = Blueprint("start", __name__)


@start_bp.route("/start", methods=["GET", "POST"])
@rate_limit(limit=40, window=60)
def start():
    if request.method != "POST":
        return render_template("start.html", concerns=CONCERNS)

    described = (request.form.get("concern") or "").strip()

    # Emergencies are checked before matching and cannot be overridden by a
    # better keyword hit. Someone typing "chest pain" gets a stop sign, not a
    # questionnaire.
    emergencies = check_emergency(described)
    if emergencies:
        return render_template(
            "start.html",
            concerns=CONCERNS,
            described=described,
            emergencies=emergencies,
        ), 200

    if not described:
        return render_template(
            "start.html", concerns=CONCERNS,
            error="Tell us what's bothering you, or pick from the list below.",
        ), 400

    return render_template(
        "start.html",
        concerns=CONCERNS,
        described=described,
        matches=match_concerns(described),
    )
