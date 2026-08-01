"""Sleep-disorder classification: None, Insomnia, or Sleep Apnea."""

from flask import Blueprint, render_template, request

from app.ml.bundle import try_get_model
from app.ratelimit import rate_limit
from app.routes.support import (
    build_summary,
    collect_and_check,
    prediction_errors,
    unavailable_page,
    urgent_interstitial,
)

sleep_bp = Blueprint("sleep", __name__, url_prefix="/sleep")

# Form field -> raw name for app.ml.features.build_sleep.
#
# Occupation is absent: the healthy rate is flat across all 11 values in the
# dataset (0.656-0.727 against a 0.70 base rate), so it carries no signal, and
# the previous route defaulted every visitor to 'Nurse'.
#
# Blood pressure is now two numbers. It used to be label-encoded against eight
# memorised readings while the form offered fourteen -- so every option in the
# "High" group (140/90, 142/92, ...) raised on submit. The builder also accepts
# a "120/80" string, which is the form the CSV uses.
FORM_TO_RAW = {
    "Gender": "Gender",
    "Age": "Age",
    "SleepDuration": "Sleep Duration",
    "QualitySleep": "Quality of Sleep",
    "PhysicalActivity": "Physical Activity Level",
    "StressLevel": "Stress Level",
    "BMICategory": "BMI Category",
    "Systolic": "Systolic",
    "Diastolic": "Diastolic",
    "HeartRate": "Heart Rate",
    "DailySteps": "Daily Steps",
}

# Shown on the result page. The previous model had no healthy class at all, so
# every visitor was told they had a disorder.
ADVICE = {
    "None": "No disorder indicated. Keep a consistent sleep schedule to stay there.",
    "Insomnia": "Signs consistent with insomnia. Consider a regular bedtime, "
                "less evening screen time, and a discussion with your doctor.",
    "Sleep Apnea": "Signs consistent with sleep apnea. This is worth raising "
                   "with a doctor, particularly if you snore or wake unrested.",
}


@sleep_bp.route("/", methods=["GET", "POST"])
@rate_limit()
@prediction_errors
def predict_sleep():
    if request.method != "POST":
        return render_template("predict_sleep.html")

    model = try_get_model("sleep")
    if model is None:
        return unavailable_page("sleep")

    raw, caveats = collect_and_check(request.form, FORM_TO_RAW, model)

    interstitial = urgent_interstitial(raw, request.form)
    if interstitial is not None:
        return interstitial

    probabilities = model.proba_one(raw)
    label = max(probabilities, key=probabilities.get)
    factors = model.explain(raw)

    headline = (
        "No sleep disorder indicated" if label == "None"
        else f"{label} indicated"
    )

    return render_template(
        "result_sleep.html",
        prediction=label,
        confidence=f"{probabilities[label] * 100:.1f}",
        advice=ADVICE[label],
        probabilities=probabilities,
        metrics=model.metadata.get("metrics", {}),
        caveats=caveats,
        factors=factors,
        factor_noun="the confidence in this result",
        factor_unit="percentage points",
        summary=build_summary(
            title="Sleep disorder screening",
            headline=f"{headline} ({probabilities[label] * 100:.0f}% confidence)",
            detail=ADVICE[label],
            model_name="sleep",
            raw=raw,
            factors=factors,
            caveats=caveats,
            form=request.form,
        ),
    )
