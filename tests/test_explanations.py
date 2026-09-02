"""Result explanations and the printable visit summary.

A percentage on its own is not something anyone can take to an appointment.
These cover the two features that make a result consultable: which of your
answers moved it, and a page you can print and hand over.
"""

import json
import re

import pytest

from app.ml.bundle import get_model
from app.ml.features import brfss_age_bucket, describe_value
from app.ml.guidance import questions_for

HIGH_RISK_HEART = {
    "high_bp": "1", "high_chol": "1", "chol_check": "1", "bmi": "34",
    "smoker": "1", "stroke": "0", "diabetes": "2", "phys_activity": "0",
    "alcohol": "0", "gen_health": "4",
    "ment_health": "15", "phys_health": "20", "diff_walk": "1", "sex": "1",
    "age": "68",
}

LOW_RISK_HEART = {
    **HIGH_RISK_HEART,
    "high_bp": "0", "high_chol": "0", "bmi": "23", "smoker": "0",
    "diabetes": "0", "phys_activity": "1", "veggies": "1",
    "gen_health": "1", "ment_health": "0", "phys_health": "0",
    "diff_walk": "0", "age": "34",
}


# --------------------------------------------------------------------------
# the age-encoding bug the explanations exposed
# --------------------------------------------------------------------------

def test_age_in_years_converts_to_the_brfss_bucket():
    assert brfss_age_bucket(18) == 1
    assert brfss_age_bucket(24) == 1
    assert brfss_age_bucket(25) == 2
    assert brfss_age_bucket(29) == 2
    assert brfss_age_bucket(30) == 3
    assert brfss_age_bucket(80) == 13
    assert brfss_age_bucket(120) == 13  # clamped, not extrapolated


def test_age_actually_changes_the_heart_prediction(client):
    """Age was inert: 25 and 80 both returned 15.73%.

    The form collects years while the model was trained on 1-13 buckets, so
    every real age landed past the model's highest split and in the same leaf.
    Age is the strongest single predictor of cardiovascular risk.
    """
    def risk(years):
        body = client.post(
            "/heart_disease/", data={**LOW_RISK_HEART, "age": str(years)}
        ).get_data(as_text=True)
        return float(re.search(r"([\d.]+)% estimated risk", body).group(1))

    young, middle, old = risk(25), risk(55), risk(85)
    assert young < middle < old
    assert old > young * 5, f"age barely moved the result: {young} -> {old}"


# --------------------------------------------------------------------------
# explanations
# --------------------------------------------------------------------------

def test_explanations_appear_on_every_result_page(client):
    cases = [
        ("/heart_disease/", HIGH_RISK_HEART),
        ("/migraine/", {
            "Age": "32", "Gender": "Female", "SleepHours": "4.5",
            "WaterIntake": "1", "SkippedMeals": "Yes", "Caffeine": "5",
            "Stress": "9", "ScreenTime": "12", "PhysicalActivity": "0",
            "Menstruating": "1",
        }),
    ]
    for path, form in cases:
        body = client.post(path, data=form).get_data(as_text=True)
        assert "What drove this result" in body, path

    # The lifestyle score is a rubric, not a model, so it shows its arithmetic
    # rather than a counterfactual explanation. See tests/test_lifestyle.py.
    body = client.post("/health-score/", data={
        "Age": "62", "BMI": "34", "ExerciseFrequency": "0",
        "DietQuality": "3", "SleepHours": "4", "SmokingStatus": "4",
        "AlcoholConsumption": "5",
    }).get_data(as_text=True)
    assert "How this was calculated" in body


def test_a_high_risk_profile_blames_the_right_things(client):
    """The top factors must be things the user actually reported as bad."""
    model = get_model("heart")
    raw = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 38, "Smoker": 1,
        "Stroke": 0, "Diabetes": 2, "PhysActivity": 0, "HvyAlcoholConsump": 0, "GenHlth": 5, "MentHlth": 20,
        "PhysHlth": 25, "DiffWalk": 1, "Sex": 1, "Age": brfss_age_bucket(70),
    }
    fields = {f.field for f in model.explain(raw, top=5)}
    assert fields & {"GenHlth", "BMI", "HighBP", "Smoker", "Diabetes", "Age"}


def test_direction_is_not_inverted_below_fifty_percent(client):
    """Binary models must explain the positive class, not the argmax one.

    Taking the argmax inverts every direction whenever risk lands under 50%:
    a 45.6% risk reported "your high blood pressure lowered it", because it was
    silently explaining the probability of *not* having the disease.
    """
    model = get_model("heart")
    raw = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34, "Smoker": 1,
        "Stroke": 0, "Diabetes": 2, "PhysActivity": 0, "HvyAlcoholConsump": 0, "GenHlth": 4, "MentHlth": 15,
        "PhysHlth": 20, "DiffWalk": 1, "Sex": 1, "Age": brfss_age_bucket(68),
    }
    assert model.proba_one(raw)["Yes"] < 0.5, "profile no longer exercises the bug"

    by_field = {f.field: f for f in model.explain(raw, top=8)}
    for field in ("HighBP", "Smoker"):
        if field in by_field:
            assert by_field[field].direction == "raised", (
                f"{field} reported as lowering risk"
            )


def test_explanations_use_words_not_codes(client):
    body = client.post("/heart_disease/", data=HIGH_RISK_HEART).get_data(as_text=True)
    assert "high blood pressure</strong> (yes)" in body
    assert "high blood pressure</strong> (1)" not in body


def test_describe_value_renders_coded_answers():
    assert describe_value("HighBP", 1) == "yes"
    assert describe_value("GenHlth", 4) == "fair"
    assert describe_value("Sex", 0) == "female"
    assert describe_value("Age", 10, "heart") == "65-69"
    # Age is plain years everywhere else, so the bucket labels must not apply.
    assert describe_value("Age", 10, "sleep") == "10"


def test_explanation_costs_one_extra_prediction(client):
    """Guards against a per-field predict loop creeping back in."""
    model = get_model("heart")
    raw = {
        "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28, "Smoker": 0,
        "Stroke": 0, "Diabetes": 0, "PhysActivity": 1, "HvyAlcoholConsump": 0, "GenHlth": 3, "MentHlth": 2,
        "PhysHlth": 2, "DiffWalk": 0, "Sex": 0, "Age": 8,
    }
    calls = []
    original = model.model.predict_proba
    model.model.predict_proba = lambda X: (calls.append(len(X)), original(X))[1]
    try:
        model.explain(raw)
    finally:
        model.model.predict_proba = original
    # One call to pick the class (1 row) plus one batched call for all variants.
    assert len(calls) <= 2
    assert max(calls) > 1, "variants were not batched"


# --------------------------------------------------------------------------
# visit summary
# --------------------------------------------------------------------------

SUBGROUP_LINE = re.compile(
    r"How well this works for ([^<]+):</strong>\s*"
    r"across ([\d,]+) people in that group, the estimate\s*"
    r"(.*?) &mdash; it predicted\s*([\d.]+)% on average where ([\d.]+)%",
    re.S,
)


@pytest.mark.parametrize("sex", ["0", "1"])
@pytest.mark.parametrize("age", ["22", "40", "58", "70", "84"])
def test_the_subgroup_line_never_contradicts_its_own_numbers(client, sex, age):
    """The result page tells the reader how the model does for their own group.

    That claim sits directly beside the two percentages it is drawn from, so it
    has to agree with them *as printed*. It briefly did not: in the youngest
    band the model is out by 6% of a 0.8% risk, which is a real ratio and an
    invisible difference, so the page said "tends to run high" next to
    "predicted 0.8% where 0.8% had it". A reader is right to trust the figures
    over the sentence, which makes the sentence the bug.
    """
    body = client.post(
        "/heart_disease/", data={**HIGH_RISK_HEART, "sex": sex, "age": age}
    ).get_data(as_text=True)

    match = SUBGROUP_LINE.search(body)
    assert match, f"no subgroup line for sex={sex} age={age}"
    _, _, direction, predicted, observed = match.groups()
    direction = re.sub(r"\s+", " ", direction).strip()
    predicted, observed = float(predicted), float(observed)

    expected = (
        "tends to run low" if observed > predicted
        else "tends to run high" if observed < predicted
        else "has been accurate on average"
    )
    assert direction == expected, (
        f"page says the estimate {direction!r} while printing "
        f"predicted {predicted}% against observed {observed}%"
    )


def test_the_subgroup_line_reads_the_right_cell(client):
    """A page quoting another group's numbers at somebody would be worse than
    quoting none, and the sex/age lookup is exactly where that goes wrong."""
    from app.ml.bundle import get_model

    cells = get_model("heart").metadata["subgroups"]["sex_age"]
    body = client.post(
        "/heart_disease/", data={**HIGH_RISK_HEART, "sex": "1", "age": "58"}
    ).get_data(as_text=True)

    match = SUBGROUP_LINE.search(body)
    assert match, "no subgroup line rendered"
    group, n, _, predicted, observed = match.groups()
    expected = cells["Male 50-64"]

    assert group == "men aged 50–64"
    assert n == f"{expected['n']:,}"
    assert float(observed) == round(expected["observed"] * 100, 1)
    assert float(predicted) == round(expected["predicted"] * 100, 1)


def _summary(client, path, form):
    body = client.post(path, data=form).get_data(as_text=True)
    match = re.search(
        r'<script type="application/json" id="assessmentSummary">(.*?)</script>',
        body, re.S,
    )
    assert match, f"{path} embedded no summary payload"
    return json.loads(match.group(1))


def test_summary_payload_is_complete(client):
    summary = _summary(client, "/heart_disease/", HIGH_RISK_HEART)
    for key in ("title", "date", "headline", "detail", "inputs", "factors",
                "caveats", "flags", "questions"):
        assert key in summary
    assert summary["inputs"], "the summary must record what was entered"
    assert summary["questions"], "the summary must suggest what to ask"


def test_summary_headline_carries_population_context(client):
    """A number without a baseline is not interpretable.

    Checks that the detail names *which* population it is comparing against,
    reading the name from the model's own metadata rather than a literal. This
    used to assert the word "population" appeared, which broke the moment the
    comparator became specific -- "among US adults" is the improvement, and the
    test was measuring the wrong thing.
    """
    population = get_model("heart").metadata["weighting"]["population"]
    summary = _summary(client, "/heart_disease/", HIGH_RISK_HEART)
    assert "%" in summary["headline"]
    assert population in summary["detail"]
    # A comparison needs both numbers: the person's, and the group's.
    assert re.search(r"\d+(\.\d+)?%", summary["detail"])


def test_questions_are_questions_not_instructions():
    """This app can help someone ask; it is not qualified to tell."""
    model = get_model("heart")
    raw = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34, "Smoker": 1,
        "Stroke": 0, "Diabetes": 2, "PhysActivity": 0, "HvyAlcoholConsump": 0, "GenHlth": 4, "MentHlth": 15,
        "PhysHlth": 20, "DiffWalk": 1, "Sex": 1, "Age": brfss_age_bucket(68),
    }
    questions = questions_for("heart", "45.6% estimated risk", model.explain(raw))
    assert questions
    banned = ("you should take", "start taking", "stop taking", "you must")
    for question in questions:
        assert not any(phrase in question.lower() for phrase in banned), question


def test_red_flags_lead_the_question_list():
    from app.ml.safety import check_red_flags

    flags = check_red_flags({"Systolic": 190, "Diastolic": 125})
    questions = questions_for("heart", "20% risk", [], flags=flags)
    assert "first" in questions[0].lower()


def test_caveats_produce_a_question_about_applicability(client):
    """If the model admits it can't judge you, that belongs in the summary."""
    summary = _summary(client, "/migraine/", {
        "Age": "80", "Gender": "Female", "SleepHours": "6.5",
        "WaterIntake": "5", "SkippedMeals": "No", "Caffeine": "2",
        "Stress": "5", "ScreenTime": "8", "PhysicalActivity": "0",
        "Menstruating": "2",
    })
    assert summary["caveats"]
    assert any("outside the range" in q or "may not apply" in q
               for q in summary["questions"])


def test_summary_page_renders_and_stores_nothing_serverside(client):
    """The page is a shell; the data lives only in the browser."""
    response = client.get("/summary")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "summaryRoot" in body
    assert "Nothing saved yet" in body
    # No result data may be baked into the server-rendered page.
    assert "estimated risk" not in body


def test_save_control_is_offered_on_every_result_page(client):
    cases = [
        ("/heart_disease/", HIGH_RISK_HEART),
        ("/migraine/", {
            "Age": "32", "Gender": "Female", "SleepHours": "4.5",
            "WaterIntake": "1", "SkippedMeals": "Yes", "Caffeine": "5",
            "Stress": "9", "ScreenTime": "12", "PhysicalActivity": "0",
            "Menstruating": "1",
        }),
    ]
    for path, form in cases:
        body = client.post(path, data=form).get_data(as_text=True)
        assert "Add to visit summary" in body, path
        assert "nothing is uploaded" in body.lower(), path
