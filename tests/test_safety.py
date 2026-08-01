"""The safety net: what the app refuses to answer, and what it interrupts for.

Before this existed, submitting a blood pressure of 190/125 with a resting
heart rate of 0 returned a calm "No Sleep Disorder" -- a reassuring answer to
input that is either a typo or an emergency, and reassuring under both.

Three tiers, in precedence order:
  1. impossible   -> 400, naming the field
  2. red flag     -> interstitial before any result
  3. out of range -> the result, plus a visible caveat

The health calculator is the vehicle for tiers 1 and 2 because it collects the
widest set of vitals. Tier 3 only applies to the two remaining models, so it is
exercised through migraine.
"""

import pytest

from app.ml.bundle import MODEL_NAMES, load_metadata
from app.ml.safety import (
    ImpossibleValue,
    check_possible,
    check_red_flags,
    check_training_range,
)

CALCULATOR = {
    "gender": "Male", "age": "40", "activity": "moderate", "height": "175",
    "weight": "70", "waist": "85", "hip": "95", "systolic": "120",
    "diastolic": "80", "water_intake": "2", "smokes_per_day": "0",
}

MIGRAINE = {
    "Age": "32", "Gender": "Female", "SleepHours": "6.5", "WaterIntake": "5",
    "SkippedMeals": "No", "Caffeine": "2", "Stress": "5", "ScreenTime": "8",
    "PhysicalActivity": "0", "Menstruating": "1",
}

SLEEP = {
    "snoring": "1", "gasping": "0", "sleepiness": "0", "insomnia_nights": "1",
    "insomnia_months": "0", "insomnia_impact": "0", "sleep_hours": "7.5",
}


def _kind(response):
    body = response.get_data(as_text=True)
    if "show my results anyway" in body:
        return "urgent"
    if "outside this model" in body:
        return "caveat"
    if response.status_code == 400:
        return "rejected"
    return "result"


# --------------------------------------------------------------------------
# tier 1 -- impossible input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,expected", [
    ("systolic", "900", "Systolic"),
    ("diastolic", "5", "Diastolic"),
    ("age", "300", "Age"),
    ("water_intake", "99", "Water intake"),
])
def test_impossible_input_is_rejected_not_answered(client, field, value, expected):
    """Guessing from a typo is worse than declining -- the user may act on it."""
    response = client.post("/health/result", data={**CALCULATOR, field: value})
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert expected.lower() in body.lower()
    assert "outside the possible range" in body


def test_the_exact_case_that_used_to_pass_silently(client):
    """BP 190/125 with a heart rate of 0 previously returned "No Sleep Disorder".

    Nothing collects a resting heart rate now, so the tier-1 half is asserted
    directly and the tier-2 half through the form that does take blood pressure.
    """
    with pytest.raises(ImpossibleValue, match="Resting heart rate"):
        check_possible("sleep", {"Heart Rate": 0})

    crisis = client.post(
        "/health/result", data={**CALCULATOR, "systolic": "190", "diastolic": "125"}
    )
    assert _kind(crisis) == "urgent"


def test_heart_bmi_500_is_rejected(client):
    heart = {
        "high_bp": "0", "high_chol": "0", "chol_check": "1", "bmi": "500",
        "smoker": "0", "stroke": "0", "diabetes": "0", "phys_activity": "1",
        "fruit": "1", "veggies": "1", "alcohol": "0", "gen_health": "1",
        "ment_health": "0", "phys_health": "0", "diff_walk": "0", "sex": "0",
        "age": "40",
    }
    response = client.post("/heart_disease/", data=heart)
    assert response.status_code == 400
    assert "estimated risk" not in response.get_data(as_text=True)


def test_check_possible_is_unit_testable():
    with pytest.raises(ImpossibleValue) as excinfo:
        check_possible("sleep", {"Heart Rate": 0})
    assert "Resting heart rate" in str(excinfo.value)
    check_possible("sleep", {"Heart Rate": 68})  # must not raise


# --------------------------------------------------------------------------
# tier 2 -- red flags
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,expected_key", [
    ({"systolic": "190", "diastolic": "125"}, "bp_crisis"),
    ({"systolic": "150", "diastolic": "95"}, "bp_stage2"),
])
def test_red_flags_interrupt_before_the_result(client, overrides, expected_key):
    response = client.post("/health/result", data={**CALCULATOR, **overrides})
    assert response.status_code == 200
    assert _kind(response) == "urgent"


def test_red_flag_can_be_acknowledged_to_continue(client):
    """The user is never blocked -- but never skips past it silently either."""
    urgent = {**CALCULATOR, "systolic": "190", "diastolic": "125"}
    assert _kind(client.post("/health/result", data=urgent)) == "urgent"

    acknowledged = client.post("/health/result", data={**urgent, "acknowledged": "1"})
    assert acknowledged.status_code == 200
    assert _kind(acknowledged) != "urgent"


def test_the_sleep_page_also_flags_blood_pressure(client):
    """Blood pressure is optional there, but checked when supplied."""
    assert _kind(client.post("/sleep/", data=SLEEP)) == "result"
    flagged = client.post(
        "/sleep/", data={**SLEEP, "systolic": "195", "diastolic": "128"}
    )
    assert _kind(flagged) == "urgent"


def test_hypertensive_crisis_wording_is_actionable():
    flags = check_red_flags({"Systolic": 190, "Diastolic": 125})
    assert flags and flags[0].key == "bp_crisis"
    assert flags[0].urgency == "emergency"
    detail = flags[0].detail.lower()
    assert "now" in detail or "today" in detail
    assert "emergency services" in detail


@pytest.mark.parametrize("values,key", [
    ({"Heart Rate": 35}, "bradycardia"),
    ({"Heart Rate": 130}, "tachycardia"),
    ({"BMI": 14}, "bmi_low"),
    ({"BMI": 45}, "bmi_high"),
])
def test_the_other_red_flag_rules(values, key):
    assert any(f.key == key for f in check_red_flags(values))


def test_emergencies_sort_above_non_emergencies():
    flags = check_red_flags({"Systolic": 190, "Diastolic": 125, "BMI": 45})
    assert flags[0].urgency == "emergency"


def test_normal_values_raise_no_flags():
    assert check_red_flags(
        {"Systolic": 118, "Diastolic": 76, "Heart Rate": 68, "BMI": 22}
    ) == []


# --------------------------------------------------------------------------
# tier 3 -- outside the training range
# --------------------------------------------------------------------------

def test_out_of_range_input_gets_the_result_plus_a_caveat(client):
    """Migraine was trained on 18-74 year olds. An 80-year-old is a guess."""
    response = client.post("/migraine/", data={**MIGRAINE, "Age": "80"})
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "outside this model" in body
    assert "18" in body and "74" in body       # the trained range is quoted


def test_in_range_input_gets_no_caveat(client):
    body = client.post("/migraine/", data=MIGRAINE).get_data(as_text=True)
    assert "outside this model" not in body


def test_check_training_range_reads_the_saved_profile():
    profile = load_metadata("migraine")["raw_profile"]
    assert check_training_range({"Age": 40}, profile) == []

    caveats = check_training_range({"Age": 95}, profile)
    assert len(caveats) == 1
    assert "unreliable" in caveats[0].message


def test_categorical_answers_are_not_caveated():
    """Forms post codes, training data holds labels -- comparing them cries wolf.

    The migraine form posts ``Menstruating=1`` where the training data reads
    "Yes". Flagging that as unseen puts a scary warning on an ordinary answer.
    """
    profile = {"BMI Category": {"kind": "categorical", "values": ["Normal", "Obese"]}}
    assert check_training_range({"BMI Category": "Normal"}, profile) == []
    assert check_training_range({"BMI Category": "1"}, profile) == []


def test_a_valid_form_submission_produces_no_caveats(client):
    """Every form's ordinary encoding must pass without a spurious warning."""
    heart = {
        "high_bp": "0", "high_chol": "0", "chol_check": "1", "bmi": "24",
        "smoker": "0", "stroke": "0", "diabetes": "0", "phys_activity": "1",
        "fruit": "1", "veggies": "1", "alcohol": "0", "gen_health": "2",
        "ment_health": "2", "phys_health": "2", "diff_walk": "0", "sex": "0",
        "age": "45",
    }
    for path, form in [("/heart_disease/", heart), ("/migraine/", MIGRAINE)]:
        body = client.post(path, data=form).get_data(as_text=True)
        assert "outside this model" not in body, f"{path} raised a spurious caveat"


# --------------------------------------------------------------------------
# the profile the whole tier-3 mechanism depends on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_records_its_raw_input_ranges(name):
    profile = load_metadata(name).get("raw_profile")
    assert profile, f"{name} has no raw_profile; retrain with ml_model/train_all.py"
    for stats in profile.values():
        if stats["kind"] == "numeric":
            assert stats["min"] <= stats["median"] <= stats["max"]
        else:
            assert stats["values"]
