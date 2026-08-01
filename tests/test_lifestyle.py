"""The lifestyle score: a rubric, and the reasons it replaced a model.

The old model was a RandomForest fit to data/synthetic_health_data.csv, which
is Gaussian noise around a formula with no clamping -- 70 rows had negative
alcohol consumption and one respondent was 1.1 years old. Worse, the form's
encodings did not match it: the diet dropdown offers 1-9 while training saw
19.9-110.3, so "9 - Excellent" landed below the worst diet the model had seen.
A maximally healthy profile scored 68.9 out of 100.
"""

import re

import pytest

from app.ml.lifestyle import FACTORS, score_lifestyle

BEST = {
    "Smoking_Status": 0, "Exercise_Frequency": 6, "BMI": 22.0,
    "Sleep_Hours": 8.0, "Alcohol_Consumption": 0, "Diet_Quality": 9,
}

WORST = {
    "Smoking_Status": 4, "Exercise_Frequency": 0, "BMI": 42.0,
    "Sleep_Hours": 4.0, "Alcohol_Consumption": 5, "Diet_Quality": 1,
}

FORM_BEST = {
    "Age": "35", "BMI": "22", "ExerciseFrequency": "6", "DietQuality": "9",
    "SleepHours": "8", "SmokingStatus": "0", "AlcoholConsumption": "0",
}

# BMI 38 rather than 42: at 40 the red-flag interstitial fires before any
# result renders, which is correct behaviour but not what these tests are for.
# test_a_red_flag_still_interrupts covers that path.
FORM_WORST = {
    "Age": "62", "BMI": "38", "ExerciseFrequency": "0", "DietQuality": "1",
    "SleepHours": "4", "SmokingStatus": "4", "AlcoholConsumption": "5",
}


# --------------------------------------------------------------------------
# the bug that motivated the replacement
# --------------------------------------------------------------------------

def test_a_perfect_profile_scores_100(client):
    """The old model gave this exact profile 68.9 and a "your diet is outside
    the training range" caveat, because the form's 1-9 diet scale sat below the
    19.9 minimum the model was trained on."""
    result = score_lifestyle(BEST)
    assert result.total == 100.0

    body = client.post("/health-score/", data=FORM_BEST).get_data(as_text=True)
    assert "100" in body
    assert "outside this model" not in body


def test_the_scale_actually_spans_its_range():
    assert score_lifestyle(WORST).total < 25
    assert score_lifestyle(BEST).total == 100


def test_every_answer_moves_the_score_in_the_right_direction():
    baseline = score_lifestyle(BEST).total
    for field, worse in [
        ("Smoking_Status", 4), ("Exercise_Frequency", 0), ("BMI", 40.0),
        ("Sleep_Hours", 4.0), ("Alcohol_Consumption", 5), ("Diet_Quality", 1),
    ]:
        degraded = score_lifestyle({**BEST, field: worse}).total
        assert degraded < baseline, f"{field} did not lower the score"


# --------------------------------------------------------------------------
# it must be transparent, because that is the whole point
# --------------------------------------------------------------------------

def test_weights_sum_to_one_hundred():
    assert sum(weight for _, _, weight, _, _ in FACTORS) == 100


def test_components_account_for_the_total_exactly():
    """A rubric that doesn't add up is no better than a model."""
    result = score_lifestyle({**BEST, "Smoking_Status": 2, "Sleep_Hours": 5.5})
    assert round(sum(c.earned for c in result.components), 1) == result.total


def test_every_component_explains_itself():
    for component in score_lifestyle(WORST).components:
        assert component.answer, f"{component.label} does not say what was entered"
        assert component.verdict, f"{component.label} gives no verdict"
        assert component.guidance, f"{component.label} cites no guidance"


def test_the_result_page_shows_the_arithmetic(client):
    body = client.post("/health-score/", data=FORM_WORST).get_data(as_text=True)
    assert "How this was calculated" in body
    for _, label, weight, _, _ in FACTORS:
        assert label in body, f"{label} is missing from the breakdown"
        assert f"/ {weight} points" in body


def test_the_page_admits_the_weights_are_a_judgement(client):
    """Presenting an editorial choice as objective is the failure being avoided."""
    body = client.post("/health-score/", data=FORM_BEST).get_data(as_text=True)
    assert "editorial judgement" in body
    assert "not fitted to" in body


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------

def test_smoking_is_weighted_highest():
    """It is the largest modifiable factor, so it should cost the most."""
    weights = {key: weight for key, _, weight, _, _ in FACTORS}
    assert weights["smoking"] == max(weights.values())


def test_age_is_not_scored():
    """Scoring someone down for getting older tells them nothing useful."""
    fields = {field for *_, field in FACTORS}
    assert "Age" not in fields
    assert score_lifestyle({**BEST, "Age": 20}).total == score_lifestyle({**BEST, "Age": 90}).total


def test_sleep_penalises_too_much_as_well_as_too_little():
    normal = score_lifestyle(BEST).total
    assert score_lifestyle({**BEST, "Sleep_Hours": 4}).total < normal
    assert score_lifestyle({**BEST, "Sleep_Hours": 12}).total < normal


def test_biggest_opportunity_points_at_the_worst_factor():
    """A heavy smoker who is otherwise healthy should be told about smoking."""
    result = score_lifestyle({**BEST, "Smoking_Status": 4})
    assert result.biggest_opportunity.key == "smoking"
    # And a perfect profile has nothing to suggest.
    assert score_lifestyle(BEST).biggest_opportunity is None


def test_missing_input_raises_rather_than_defaulting():
    incomplete = {k: v for k, v in BEST.items() if k != "Smoking_Status"}
    with pytest.raises(ValueError, match="Smoking_Status"):
        score_lifestyle(incomplete)


def test_the_form_still_validates_and_flags(client):
    """The rubric route keeps the safety net the model route had."""
    partial = {"Age": "35", "BMI": "22"}
    assert client.post("/health-score/", data=partial).status_code == 400
    assert client.post(
        "/health-score/", data={**FORM_BEST, "BMI": "500"}
    ).status_code == 400


def test_a_red_flag_still_interrupts(client):
    """BMI 40+ is WHO class III and interrupts before any score is shown."""
    response = client.post("/health-score/", data={**FORM_BEST, "BMI": "42"})
    body = response.get_data(as_text=True)
    assert "show my results anyway" in body
    assert "How this was calculated" not in body

    proceeded = client.post(
        "/health-score/", data={**FORM_BEST, "BMI": "42", "acknowledged": "1"}
    )
    assert "How this was calculated" in proceeded.get_data(as_text=True)


def test_it_still_produces_a_visit_summary(client):
    import json

    body = client.post("/health-score/", data=FORM_WORST).get_data(as_text=True)
    match = re.search(
        r'<script type="application/json" id="assessmentSummary">(.*?)</script>',
        body, re.S,
    )
    assert match
    summary = json.loads(match.group(1))
    assert summary["title"] == "Lifestyle score"
    assert summary["questions"]
    assert summary["caveats"] == [], "a rubric has nothing to extrapolate beyond"
