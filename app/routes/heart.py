"""Heart-disease risk, from the BRFSS 2015 indicators model."""

from flask import Blueprint, render_template, request

from app.ml.bundle import try_get_model
from app.ml.features import brfss_age_bucket
from app.ratelimit import rate_limit
from app.routes.support import (
    build_summary,
    collect_and_check,
    prediction_errors,
    unavailable_page,
    urgent_interstitial,
)

heart_disease_bp = Blueprint("heart_disease", __name__, url_prefix="/heart_disease")

# Form field -> the raw name app.ml.features.build_heart expects. All feature
# engineering happens in that builder, which the training script also uses.
#
# Education and Income are absent because the form never asked for them. The
# previous version of this route hardcoded 4 and 5 for every visitor.
FORM_TO_RAW = {
    "high_bp": "HighBP",
    "high_chol": "HighChol",
    "chol_check": "CholCheck",
    "bmi": "BMI",
    "smoker": "Smoker",
    "stroke": "Stroke",
    "diabetes": "Diabetes",
    "phys_activity": "PhysActivity",
    "fruit": "Fruits",
    "veggies": "Veggies",
    "alcohol": "HvyAlcoholConsump",
    "gen_health": "GenHlth",
    "ment_health": "MentHlth",
    "phys_health": "PhysHlth",
    "diff_walk": "DiffWalk",
    "sex": "Sex",
    "age": "Age",
}


def _to_model_units(values):
    """The form asks for age in years; the model was trained on BRFSS buckets.

    Passing years straight through -- which is what used to happen -- put every
    real age past the model's highest split, so a 25-year-old and an
    80-year-old received the same 15.73% risk. Age was inert.
    """
    return {**values, "Age": brfss_age_bucket(values["Age"])}


@heart_disease_bp.route("/", methods=["GET", "POST"])
@rate_limit()
@prediction_errors
def predict_heart_disease():
    if request.method != "POST":
        return render_template("predict_heart.html")

    model = try_get_model("heart")
    if model is None:
        return unavailable_page("heart")

    raw, caveats = collect_and_check(
        request.form, FORM_TO_RAW, model, transform=_to_model_units
    )

    interstitial = urgent_interstitial(raw, request.form)
    if interstitial is not None:
        return interstitial

    probability = model.proba_one(raw)["Yes"]

    # The threshold comes from the model, not a hardcoded 0.5. Only 9.4% of the
    # training population has heart disease, so a calibrated model rarely exceeds
    # 0.5 for anyone; the stored threshold is tuned on validation data to balance
    # missed cases against false alarms.
    threshold = model.metadata.get("decision_threshold", 0.5)
    factors = model.explain(raw)
    metrics = model.metadata.get("metrics", {})
    prevalence = metrics.get("observed_prevalence", 0) * 100

    return render_template(
        "result_heart.html",
        prediction="Yes" if probability >= threshold else "No",
        probability=f"{probability * 100:.2f}",
        threshold=f"{threshold * 100:.1f}",
        metrics=metrics,
        caveats=caveats,
        factors=factors,
        factor_noun="your estimated risk",
        factor_unit="percentage points",
        summary=build_summary(
            title="Heart disease risk",
            headline=f"{probability * 100:.1f}% estimated risk",
            detail=(
                f"Compared with {prevalence:.1f}% across the survey population. "
                f"This assessment flags anything above {threshold * 100:.1f}% "
                f"for follow-up."
            ),
            model_name="heart",
            raw=raw,
            factors=factors,
            caveats=caveats,
            form=request.form,
        ),
    )
