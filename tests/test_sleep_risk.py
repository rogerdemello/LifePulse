"""Sleep screening: observed national rates, and an insomnia criteria check.

The old model was trained on a file whose respondents all had a systolic blood
pressure between 110 and 144 and a resting pulse between 60 and 89, so the app
refused to trust its own answer for anyone hypertensive -- exactly the people
most likely to have sleep apnea. It also had two classes and could not tell
anyone they were fine.

Retraining on NHANES 2017-18 fixed the range and then showed no model was
warranted: snoring plus daytime sleepiness reaches ROC-AUC 0.791, adding age,
sex, BMI, blood pressure and pulse drops it to 0.741, and an unfitted rule
matches the fitted model. The page reports the survey's own numbers instead.
"""

import pytest

from app.ml.sleep_risk import (
    NHANES_APNEA_RATES,
    NHANES_OVERALL_RATE,
    NHANES_SAMPLE,
    assess_apnea,
    assess_insomnia,
)

HEALTHY = {
    "snoring": "0", "gasping": "0", "sleepiness": "0",
    "insomnia_nights": "0", "insomnia_months": "0", "insomnia_impact": "0",
    "sleep_hours": "8",
}

CONCERNING = {
    "snoring": "3", "gasping": "0", "sleepiness": "2",
    "insomnia_nights": "5", "insomnia_months": "1", "insomnia_impact": "1",
    "sleep_hours": "5",
}


# --------------------------------------------------------------------------
# the lookup table
# --------------------------------------------------------------------------

def test_the_table_is_complete_and_plausible():
    for snoring in range(4):
        for sleepiness in range(3):
            rate, sample = NHANES_APNEA_RATES[(snoring, sleepiness)]
            assert 0 < rate < 1
            # Every cell must hold enough people to quote a percentage from.
            assert sample >= 150, f"cell ({snoring},{sleepiness}) has only {sample}"
    assert sum(n for _, n in NHANES_APNEA_RATES.values()) == NHANES_SAMPLE


def test_risk_rises_with_both_questions():
    """Monotonic in each direction, which is what makes the table readable."""
    for sleepiness in range(3):
        rates = [NHANES_APNEA_RATES[(s, sleepiness)][0] for s in range(4)]
        assert rates == sorted(rates), f"not monotonic in snoring at {sleepiness}"
    for snoring in range(4):
        rates = [NHANES_APNEA_RATES[(snoring, s)][0] for s in range(3)]
        assert rates == sorted(rates), f"not monotonic in sleepiness at {snoring}"


def test_the_extremes_are_far_apart():
    """If the two questions barely separated anyone, this page would be noise."""
    lowest = NHANES_APNEA_RATES[(0, 0)][0]
    highest = NHANES_APNEA_RATES[(3, 2)][0]
    assert highest >= 10 * lowest


# --------------------------------------------------------------------------
# apnea assessment
# --------------------------------------------------------------------------

def test_a_low_risk_profile_is_told_so():
    result = assess_apnea(snoring=0, sleepiness=0)
    assert result.band == "low"
    assert result.percent <= 5
    assert "least likely" in result.headline


def test_a_high_risk_profile_is_told_so():
    result = assess_apnea(snoring=3, sleepiness=2)
    assert result.band == "high"
    assert result.percent >= 20
    assert result.times_average > 2


def test_witnessed_gasping_overrides_the_statistic():
    """It is the symptom the survey counted, so quoting a probability of
    reporting it back at someone who just reported it would be absurd."""
    result = assess_apnea(snoring=0, sleepiness=0, witnessed_gasping=True)
    assert result.band == "high"
    assert result.witnessed_gasping
    assert "already noticed" in result.headline
    assert "doctor" in result.comparison


def test_every_result_says_where_the_number_came_from():
    for snoring in range(4):
        for sleepiness in range(3):
            result = assess_apnea(snoring, sleepiness)
            assert str(result.sample) in result.comparison.replace(",", "")
            assert "NHANES" in result.comparison


def test_out_of_range_answers_are_clamped_not_extrapolated():
    """There is no training range here, but a tampered form must still be safe."""
    assert assess_apnea(99, 99).band == assess_apnea(3, 2).band
    assert assess_apnea(-5, -5).band == assess_apnea(0, 0).band


def test_the_overall_rate_matches_the_table():
    weighted = sum(rate * n for rate, n in NHANES_APNEA_RATES.values())
    total = sum(n for _, n in NHANES_APNEA_RATES.values())
    assert abs(weighted / total - NHANES_OVERALL_RATE) < 0.015


# --------------------------------------------------------------------------
# insomnia criteria
# --------------------------------------------------------------------------

def test_all_three_criteria_are_required():
    assert assess_insomnia(5, True, True).meets_criteria
    assert not assess_insomnia(2, True, True).meets_criteria     # too few nights
    assert not assess_insomnia(5, False, True).meets_criteria    # too recent
    assert not assess_insomnia(5, True, False).meets_criteria    # no daytime effect


def test_recent_but_frequent_trouble_is_described_honestly():
    result = assess_insomnia(5, False, True)
    assert not result.meets_criteria
    assert "not yet for three months" in result.summary


def test_no_trouble_is_reported_plainly():
    assert "No trouble sleeping" in assess_insomnia(0, False, False).summary


def test_meeting_the_criteria_points_at_treatment():
    detail = assess_insomnia(5, True, True).detail
    assert "CBT-I" in detail or "talking therapy" in detail


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------

def test_a_healthy_profile_gets_a_reassuring_page(client):
    body = client.post("/sleep/", data=HEALTHY).get_data(as_text=True)
    assert "least likely to have sleep apnea" in body
    assert "No trouble sleeping reported" in body


def test_a_concerning_profile_is_told_to_get_assessed(client):
    body = client.post("/sleep/", data=CONCERNING).get_data(as_text=True)
    assert "most likely" in body
    assert "sleep study" in body
    assert "chronic insomnia" in body


def test_the_page_shows_its_source(client):
    body = client.post("/sleep/", data=HEALTHY).get_data(as_text=True)
    assert "NHANES" in body
    assert "No model is involved" in body


def test_no_extrapolation_caveat_is_ever_shown(client):
    """There is no training range to fall outside any more."""
    for form in (HEALTHY, CONCERNING):
        body = client.post("/sleep/", data=form).get_data(as_text=True)
        assert "outside this model" not in body


def test_blood_pressure_is_optional(client):
    assert client.post("/sleep/", data=HEALTHY).status_code == 200
    with_bp = {**HEALTHY, "systolic": "118", "diastolic": "76"}
    assert client.post("/sleep/", data=with_bp).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("snoring", "9"), ("sleepiness", "-1"), ("insomnia_nights", "20"),
    ("gasping", "maybe"), ("sleep_hours", "lots"),
])
def test_tampered_choices_are_rejected(client, field, value):
    response = client.post("/sleep/", data={**HEALTHY, field: value})
    assert response.status_code == 400
    assert "Traceback" not in response.get_data(as_text=True)


def test_it_produces_a_visit_summary(client):
    import json
    import re

    body = client.post("/sleep/", data=CONCERNING).get_data(as_text=True)
    match = re.search(
        r'<script type="application/json" id="assessmentSummary">(.*?)</script>',
        body, re.S,
    )
    assert match
    summary = json.loads(match.group(1))
    assert summary["title"] == "Sleep screening"
    assert summary["questions"]
    assert summary["caveats"] == []
    assert any("sleep study" in q for q in summary["questions"])
