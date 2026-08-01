"""Lifestyle score (0-100) from a transparent rubric.

This route used to serve a RandomForest trained on synthetic data whose
encodings did not match this form's, so a user selecting "9 - Excellent" for
diet was scored below the worst diet the model had ever seen. See
``app/ml/lifestyle.py`` for the full account and the replacement.

There is no model here now, so there is nothing to caveat, nothing to
extrapolate beyond, and every point is traceable to a stated rule.
"""

from flask import Blueprint, render_template, request

from app.ml.lifestyle import score_lifestyle
from app.ml.safety import check_possible
from app.ratelimit import rate_limit
from app.routes.support import (
    build_rubric_summary,
    collect,
    prediction_errors,
    urgent_interstitial,
)

health_score_bp = Blueprint("health_score", __name__, url_prefix="/health-score")

FORM_TO_RAW = {
    "Age": "Age",
    "BMI": "BMI",
    "ExerciseFrequency": "Exercise_Frequency",
    "DietQuality": "Diet_Quality",
    "SleepHours": "Sleep_Hours",
    "SmokingStatus": "Smoking_Status",
    "AlcoholConsumption": "Alcohol_Consumption",
}


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


@health_score_bp.route("/", methods=["GET", "POST"])
@rate_limit()
@prediction_errors
def predict_health_score():
    if request.method != "POST":
        return render_template("predict_health_score.html")

    raw = collect(request.form, FORM_TO_RAW)
    check_possible("health_score", raw)

    interstitial = urgent_interstitial(raw, request.form)
    if interstitial is not None:
        return interstitial

    result = score_lifestyle(raw)

    return render_template(
        "result_health_score.html",
        result=result,
        score=result.total,
        rating=result.band,
        color=result.colour,
        interpretation=result.interpretation,
        bmi=float(raw["BMI"]),
        bmi_cat=bmi_category(float(raw["BMI"])),
        summary=build_rubric_summary(result, raw, request.form),
    )
