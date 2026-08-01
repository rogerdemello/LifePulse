"""Migraine-risk assessment."""

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

migraine_bp = Blueprint("migraine", __name__, url_prefix="/migraine")

# Form field -> raw name for app.ml.features.build_migraine.
#
# This route used to reimplement the model's feature engineering inline, with
# different names AND different formulas than the training script -- e.g. it
# computed Dehydration_Risk as (Caffeine > 2) & (Water < 2) where training used
# (Caffeine > 3) & (Water < 4). All of it now lives in the shared builder.
FORM_TO_RAW = {
    "Age": "Age",
    "Gender": "Gender",
    "SleepHours": "Sleep Hours",
    "WaterIntake": "Water Intake",
    "SkippedMeals": "Skipped Meals",
    "Caffeine": "Caffeine",
    "Stress": "Stress",
    "ScreenTime": "Screen Time",
    "PhysicalActivity": "Physical Activity",
    "Menstruating": "Menstruating",
}


@migraine_bp.route("/", methods=["GET", "POST"])
@rate_limit()
@prediction_errors
def predict_migraine():
    if request.method != "POST":
        return render_template("predict_migraine.html")

    model = try_get_model("migraine")
    if model is None:
        return unavailable_page("migraine")

    raw, caveats = collect_and_check(request.form, FORM_TO_RAW, model)

    interstitial = urgent_interstitial(raw, request.form)
    if interstitial is not None:
        return interstitial

    probabilities = model.proba_one(raw)
    label = max(probabilities, key=probabilities.get)
    factors = model.explain(raw)
    risk = probabilities["Migraine Risk"] * 100

    return render_template(
        "result_migraine.html",
        prediction=label,
        confidence=round(probabilities[label] * 100, 1),
        caveats=caveats,
        factors=factors,
        factor_noun="your estimated migraine risk",
        factor_unit="percentage points",
        summary=build_summary(
            title="Migraine risk",
            headline=f"{label} ({risk:.0f}% estimated risk)",
            detail="Based on 10 lifestyle answers and 10 derived interactions.",
            model_name="migraine",
            raw=raw,
            factors=factors,
            caveats=caveats,
            form=request.form,
        ),
    )
