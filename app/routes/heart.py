"""Heart-disease risk, from the BRFSS 2015 indicators model."""

from flask import Blueprint, render_template, request

from app.ml.bundle import try_get_model
from app.ml.features import brfss_age_band, brfss_age_bucket
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
# Education and Income are absent because the form never asked for them: the
# previous version of this route hardcoded 4 and 5 for every visitor.
#
# Fruit and vegetable intake are absent because they moved the answer by
# 0.0001 ROC-AUC, and BRFSS stopped asking after 2015 -- so two questions that
# bought nothing were also what pinned the model to a decade-old survey.
FORM_TO_RAW = {
    "high_bp": "HighBP",
    "high_chol": "HighChol",
    "chol_check": "CholCheck",
    "bmi": "BMI",
    "smoker": "Smoker",
    "stroke": "Stroke",
    "diabetes": "Diabetes",
    "phys_activity": "PhysActivity",
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


def _what_happened_to_people_scored_like_you(model, probability):
    """The observed outcome rate for the reader's band, with its interval.

    The page was printing this risk to two decimal places -- 12.34% -- from a
    model whose Brier score is 0.057. That is four significant figures of a
    quantity good to about one, and a number that precise reads as a
    measurement rather than an estimate.

    Rather than model the uncertainty, report it. ``risk_bins`` in the metadata
    records, for each band of predicted risk, what share of held-out survey
    respondents in that band actually had heart disease, with a 95% interval
    around it. That is the same move app/ml/sleep_risk.py makes: an observed
    rate from real data, checkable against the published file, instead of a
    number that has to be taken on trust.
    """
    for band in model.metadata.get("risk_bins", []):
        if not band["low"] <= probability < band["high"]:
            continue

        # The band's own rate is deliberately not shown as though it were the
        # reader's. Bands are wide -- 10-20% is one of them -- so somebody
        # scored 10.3% would be told "people like you: 15%", which overstates
        # their risk by half. What transfers is the *width*: how tightly the
        # outcome rate is pinned down for people the model scores this way.
        half_width = (band["observed_high"] - band["observed_low"]) / 2
        return {
            "estimate": f"{probability * 100:.0f}",
            "low": f"{max(0.0, probability - half_width) * 100:.0f}",
            "high": f"{min(1.0, probability + half_width) * 100:.0f}",
            "give_or_take": f"{half_width * 100:.0f}",
            "n": f"{band['n']:,}",
        }
    return None


def _how_it_does_for_people_like_you(model, raw):
    """The model's measured accuracy for the reader's own sex and age band.

    An aggregate ROC-AUC of 0.855 is an average over 62,000 people, and averages
    hide their tails: this model separates cases well at 35-49 (0.86) and poorly
    past 80 (0.67), and it under-states risk by about a tenth in the 50-64 band.
    Somebody reading their own number deserves to be told which of those they
    are, in the same breath as the number.

    Returns ``None`` when the model carries no subgroup audit, or when the
    reader's cell is too small to say anything honest about -- silence is better
    than a confidence interval nobody can see.
    """
    subgroups = model.metadata.get("subgroups", {}).get("sex_age", {})
    if not subgroups:
        return None

    male = int(float(raw["Sex"])) == 1
    band = brfss_age_band(raw["Age"])
    stats = subgroups.get(f"{'Male' if male else 'Female'} {band}")
    if not stats or stats["n"] < 500:
        return None

    if stats.get("observed_over_predicted") is None:
        return None

    # Decide the direction from the numbers the page will actually print, not
    # from the full-precision ratio. In the youngest band the model is out by 6%
    # of a 0.8% risk, which is a real ratio and an invisible difference: saying
    # "tends to run high" next to "predicted 0.8% where 0.8% had it" reads as a
    # contradiction, and a reader is right to trust the figures over the claim.
    observed = round(stats["observed"] * 100, 1)
    predicted = round(stats["predicted"] * 100, 1)

    # Observed above predicted means the model under-states this group's risk,
    # so the estimate on the page "runs low". Getting this backwards would tell
    # someone to worry less precisely where they should worry more.
    if observed > predicted:
        direction = "tends to run low"
    elif observed < predicted:
        direction = "tends to run high"
    else:
        direction = "has been accurate on average"

    return {
        "group": f"{'men' if male else 'women'} aged "
                 f"{band.replace('80+', '80 and over').replace('-', '–')}",
        "direction": direction,
        "n": f"{stats['n']:,}",
        "observed": f"{observed:.1f}",
        "predicted": f"{predicted:.1f}",
        "roc_auc": f"{stats['roc_auc']:.3f}" if "roc_auc" in stats else None,
    }


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

    # What the comparison figure is a figure *about*. BRFSS is a weighted
    # survey, so its rows are not a cross-section of the country. This model is
    # scored and thresholded with those weights -- see ml_model/train_all.py for
    # why the fit deliberately isn't -- which makes the prevalence above an
    # estimate for US adults rather than for people who answer phone surveys.
    # The wording comes from the model's own metadata so that retraining on some
    # other population cannot leave this sentence behind describing the old one.
    population = model.metadata.get("weighting", {}).get(
        "population", "the survey population"
    )

    return render_template(
        "result_heart.html",
        prediction="Yes" if probability >= threshold else "No",
        # One decimal, not two. The second was four significant figures of a
        # quantity the band below says is good to about one.
        probability=f"{probability * 100:.1f}",
        band=_what_happened_to_people_scored_like_you(model, probability),
        threshold=f"{threshold * 100:.1f}",
        metrics=metrics,
        population=population,
        subgroup=_how_it_does_for_people_like_you(model, raw),
        caveats=caveats,
        factors=factors,
        factor_noun="your estimate",
        factor_unit="points in 100",
        summary=build_summary(
            title="Heart disease risk",
            headline=f"{probability * 100:.1f}% estimated risk",
            detail=(
                f"Compared with {prevalence:.1f}% among {population}. "
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
