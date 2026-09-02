"""Sleep screening: apnea signs and an insomnia criteria check.

No model. See ``app/ml/sleep_risk.py`` for why -- retraining on real national
data showed that two questions carry the whole signal, and that an unfitted
rule matches a fitted model over nine features.
"""

from flask import Blueprint, render_template, request

from app.ml.safety import check_possible
from app.ml.sleep_risk import (
    NHANES_CYCLE,
    NHANES_SAMPLE,
    SLEEPINESS_LABELS,
    SNORING_LABELS,
    assess_apnea,
    assess_insomnia,
)
from app.ratelimit import rate_limit
from app.routes.support import (
    FormError,
    build_sleep_summary,
    prediction_errors,
    urgent_interstitial,
)

sleep_bp = Blueprint("sleep", __name__, url_prefix="/sleep")

REQUIRED = ["snoring", "sleepiness", "gasping", "insomnia_nights",
            "insomnia_months", "insomnia_impact", "sleep_hours"]


@sleep_bp.route("/", methods=["GET", "POST"])
@rate_limit()
@prediction_errors
def predict_sleep():
    if request.method != "POST":
        return render_template(
            "predict_sleep.html",
            snoring_labels=SNORING_LABELS,
            sleepiness_labels=SLEEPINESS_LABELS,
        )

    form = request.form
    missing = [f for f in REQUIRED if not str(form.get(f, "")).strip()]
    if missing:
        raise FormError(
            "Please answer every question. Missing: " + ", ".join(sorted(missing))
        )

    # Every one of these is a <select> or a bounded number, so anything that
    # is not a plain integer in range did not come from the form. Reject it as
    # a bad answer rather than letting int() raise into a 500.
    choices = {
        "snoring": range(0, 4), "sleepiness": range(0, 3), "gasping": range(0, 2),
        "insomnia_nights": range(0, 8), "insomnia_months": range(0, 2),
        "insomnia_impact": range(0, 2),
    }
    for field, allowed in choices.items():
        try:
            value = int(form[field])
        except (TypeError, ValueError) as exc:
            raise FormError(
                f"“{field.replace('_', ' ')}” was not a valid choice."
            ) from exc
        if value not in allowed:
            raise FormError(
                f"“{field.replace('_', ' ')}” must be between "
                f"{allowed[0]} and {allowed[-1]}."
            )
    try:
        float(form["sleep_hours"])
    except (TypeError, ValueError) as exc:
        raise FormError("Hours of sleep must be a number.") from exc

    # Blood pressure is optional here -- it is not used for the assessment, but
    # if it is given it goes through the same safety checks as everywhere else.
    vitals = {}
    for field, name in [("systolic", "Systolic"), ("diastolic", "Diastolic"),
                        ("sleep_hours", "Sleep Duration")]:
        value = str(form.get(field, "")).strip()
        if value:
            vitals[name] = value
    check_possible("sleep", vitals)

    interstitial = urgent_interstitial(vitals, form)
    if interstitial is not None:
        return interstitial

    apnea = assess_apnea(
        snoring=form["snoring"],
        sleepiness=form["sleepiness"],
        witnessed_gasping=form["gasping"] == "1",
    )
    insomnia = assess_insomnia(
        nights_per_week=form["insomnia_nights"],
        months_3_plus=form["insomnia_months"] == "1",
        daytime_impact=form["insomnia_impact"] == "1",
    )
    sleep_hours = float(form["sleep_hours"])

    return render_template(
        "result_sleep.html",
        apnea=apnea,
        insomnia=insomnia,
        sleep_hours=sleep_hours,
        snoring_labels=SNORING_LABELS,
        sleepiness_labels=SLEEPINESS_LABELS,
        source_cycle=NHANES_CYCLE,
        source_sample=NHANES_SAMPLE,
        summary=build_sleep_summary(apnea, insomnia, sleep_hours, vitals, form),
    )
