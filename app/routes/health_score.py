"""Composite health score (0-100).

The underlying dataset is generated, not observed -- see the ``data_note`` in
``app/models/health_score/metadata.json``. Scores are relative to that synthetic
distribution (mean ~85), which is why the rating bands sit high.
"""

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

# (minimum score, rating, bootstrap colour, interpretation)
BANDS = [
    (90, "Excellent", "success", "Outstanding health! You're in the top tier."),
    (80, "Very Good", "success", "Great health profile! Keep up the excellent habits."),
    (70, "Good", "primary", "Solid health foundation with some room for improvement."),
    (60, "Fair", "warning", "Moderate health - focus on making improvements."),
    (0, "Needs Improvement", "danger",
     "Your health needs attention. Consider lifestyle changes."),
]


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

    model = try_get_model("health_score")
    if model is None:
        return unavailable_page("health_score")

    raw, caveats = collect_and_check(request.form, FORM_TO_RAW, model)

    interstitial = urgent_interstitial(raw, request.form)
    if interstitial is not None:
        return interstitial

    # predict_one runs the builder, which rejects anything non-numeric, so the
    # conversions below cannot fail once it has returned. The template compares
    # these values numerically, so they must not stay as form strings.
    score = min(100.0, max(0.0, float(model.predict_one(raw))))
    values = {key: float(value) for key, value in raw.items()}

    rating, color, interpretation = next(
        (r, c, i) for threshold, r, c, i in BANDS if score >= threshold
    )
    factors = model.explain(raw)

    return render_template(
        "result_health_score.html",
        score=round(score, 1),
        rating=rating,
        color=color,
        interpretation=interpretation,
        bmi=values["BMI"],
        bmi_cat=bmi_category(values["BMI"]),
        exercise=values["Exercise_Frequency"],
        diet=values["Diet_Quality"],
        sleep=values["Sleep_Hours"],
        smoking=values["Smoking_Status"],
        alcohol=values["Alcohol_Consumption"],
        caveats=caveats,
        factors=factors,
        factor_noun="your score",
        factor_unit="points",
        summary=build_summary(
            title="Lifestyle health score",
            headline=f"{score:.0f} out of 100 — {rating}",
            detail=(
                f"{interpretation} Note: this model is trained on synthetic data, "
                f"so treat the number as illustrative."
            ),
            model_name="health_score",
            raw=raw,
            factors=factors,
            caveats=caveats,
            form=request.form,
        ),
    )
