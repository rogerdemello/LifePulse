"""The safety net: what the app refuses to answer, and what it interrupts for.

Before this existed, submitting a blood pressure of 190/125 with a resting heart
rate of 0 returned a calm "No Sleep Disorder" -- a reassuring answer to input
that is either a typo or an emergency, and reassuring under both readings.

Three tiers, in precedence order:
  1. impossible   -> 400, naming the field
  2. red flag     -> interstitial before any model runs
  3. out of range -> the result, plus a visible caveat
"""

import pytest

from app.ml.bundle import MODEL_NAMES, load_metadata
from app.ml.safety import (
    ImpossibleValue,
    check_possible,
    check_red_flags,
    check_training_range,
)

SLEEP = {
    "Gender": "Female", "Age": "30", "SleepDuration": "7.5", "QualitySleep": "8",
    "PhysicalActivity": "60", "StressLevel": "3", "BMICategory": "Normal",
    "Systolic": "118", "Diastolic": "76", "HeartRate": "68", "DailySteps": "9000",
}

HEART = {
    "high_bp": "0", "high_chol": "0", "chol_check": "1", "bmi": "22",
    "smoker": "0", "stroke": "0", "diabetes": "0", "phys_activity": "1",
    "fruit": "1", "veggies": "1", "alcohol": "0", "gen_health": "1",
    "ment_health": "0", "phys_health": "0", "diff_walk": "0", "sex": "0",
    "age": "5",
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

@pytest.mark.parametrize("field,value,expected_in_message", [
    ("HeartRate", "0", "heart rate"),
    ("Systolic", "900", "Systolic"),
    ("SleepDuration", "25", "Sleep duration"),
    ("Age", "300", "Age"),
])
def test_impossible_input_is_rejected_not_answered(client, field, value, expected_in_message):
    """Guessing from a typo is worse than declining -- the user may act on it."""
    response = client.post("/sleep/", data={**SLEEP, field: value})
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert expected_in_message.lower() in body.lower()
    # It must say what a sensible value looks like, not just refuse.
    assert "outside the possible range" in body


def test_the_exact_case_that_used_to_pass_silently(client):
    """BP 190/125 with a heart rate of 0 previously returned 'No Sleep Disorder'."""
    response = client.post(
        "/sleep/", data={**SLEEP, "Systolic": "190", "Diastolic": "125", "HeartRate": "0"}
    )
    assert response.status_code == 400
    assert "No Sleep Disorder" not in response.get_data(as_text=True)


def test_heart_bmi_500_is_rejected(client):
    response = client.post("/heart_disease/", data={**HEART, "bmi": "500"})
    assert response.status_code == 400
    assert "estimated risk" not in response.get_data(as_text=True)


def test_age_is_collected_in_years_and_bounded_as_years(client):
    """The heart form asks for years, so 99 is a 99-year-old, not an error.

    This test previously asserted the opposite, because the route passed the
    entered years straight to a model trained on BRFSS 5-year buckets. See
    tests/test_explanations.py for the effect that had.
    """
    assert client.post("/heart_disease/", data={**HEART, "age": "99"}).status_code == 200
    assert client.post("/heart_disease/", data={**HEART, "age": "300"}).status_code == 400
    assert client.post("/heart_disease/", data={**HEART, "age": "-5"}).status_code == 400


def test_check_possible_is_unit_testable():
    with pytest.raises(ImpossibleValue) as excinfo:
        check_possible("sleep", {"Heart Rate": 0})
    assert "Resting heart rate" in str(excinfo.value)
    check_possible("sleep", {"Heart Rate": 68})  # must not raise


# --------------------------------------------------------------------------
# tier 2 -- red flags
# --------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,expected_key", [
    ({"Systolic": "190", "Diastolic": "125"}, "bp_crisis"),
    ({"Systolic": "150", "Diastolic": "95"}, "bp_stage2"),
    ({"HeartRate": "35"}, "bradycardia"),
    ({"HeartRate": "130"}, "tachycardia"),
])
def test_red_flags_interrupt_before_the_result(client, overrides, expected_key):
    response = client.post("/sleep/", data={**SLEEP, **overrides})
    assert response.status_code == 200
    assert _kind(response) == "urgent"
    # The assessment result must not be shown underneath the warning.
    assert "No Sleep Disorder" not in response.get_data(as_text=True)


def test_red_flag_can_be_acknowledged_to_continue(client):
    """The user is never blocked -- but never skips past it silently either."""
    urgent = {**SLEEP, "Systolic": "190", "Diastolic": "125"}
    assert _kind(client.post("/sleep/", data=urgent)) == "urgent"

    acknowledged = client.post("/sleep/", data={**urgent, "acknowledged": "1"})
    assert acknowledged.status_code == 200
    assert _kind(acknowledged) != "urgent"


def test_hypertensive_crisis_wording_is_actionable():
    flags = check_red_flags({"Systolic": 190, "Diastolic": 125})
    assert flags and flags[0].key == "bp_crisis"
    assert flags[0].urgency == "emergency"
    detail = flags[0].detail.lower()
    assert "now" in detail or "today" in detail
    assert "emergency services" in detail


def test_emergencies_sort_above_non_emergencies():
    flags = check_red_flags({"Systolic": 190, "Diastolic": 125, "BMI": 45})
    assert [f.urgency for f in flags] == sorted(
        [f.urgency for f in flags], key=lambda u: 0 if u == "emergency" else 1
    )
    assert flags[0].urgency == "emergency"


def test_normal_values_raise_no_flags():
    assert check_red_flags(
        {"Systolic": 118, "Diastolic": 76, "Heart Rate": 68, "BMI": 22}
    ) == []


def test_the_calculator_also_gets_the_blood_pressure_interrupt(client):
    """It collects BP, so it gets the same safety net as the models."""
    form = {
        "gender": "Male", "age": "40", "activity": "moderate", "height": "175",
        "weight": "70", "waist": "85", "hip": "95", "systolic": "190",
        "diastolic": "125", "water_intake": "2", "smokes_per_day": "0",
    }
    assert _kind(client.post("/health/result", data=form)) == "urgent"


# --------------------------------------------------------------------------
# tier 3 -- outside the training range
# --------------------------------------------------------------------------

def test_out_of_range_input_gets_the_result_plus_a_caveat(client):
    """The sleep model saw no systolic above 144. It should say so.

    This replaces an earlier test that asserted 163/104 merely returned 200 --
    true, but it was the wrong expectation: a confident answer for input the
    model has no evidence about is the problem, not the goal.
    """
    response = client.post(
        "/sleep/",
        data={**SLEEP, "Systolic": "163", "Diastolic": "104", "acknowledged": "1"},
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "outside this model" in body
    assert "110" in body and "144" in body  # the trained range is quoted


def test_in_range_input_gets_no_caveat(client):
    body = client.post("/sleep/", data=SLEEP).get_data(as_text=True)
    assert "outside this model" not in body


def test_check_training_range_reads_the_saved_profile():
    profile = load_metadata("sleep")["raw_profile"]
    assert check_training_range({"Heart Rate": 70}, profile) == []

    caveats = check_training_range({"Heart Rate": 150}, profile)
    assert len(caveats) == 1
    assert "unreliable" in caveats[0].message


def test_categorical_answers_are_not_caveated():
    """Forms post codes, training data holds labels -- comparing them cries wolf.

    The migraine form posts ``Menstruating=1`` where the training data reads
    "Yes". Flagging that as unseen puts a scary warning on a perfectly ordinary
    answer, which teaches people to scroll past the warnings that matter. The
    feature builders already reject categories they cannot interpret.
    """
    profile = {"BMI Category": {"kind": "categorical", "values": ["Normal", "Obese"]}}
    assert check_training_range({"BMI Category": "Normal"}, profile) == []
    assert check_training_range({"BMI Category": "1"}, profile) == []


def test_a_valid_form_submission_produces_no_caveats(client):
    """Every form's ordinary encoding must pass without a spurious warning."""
    cases = [
        ("/sleep/", SLEEP),
        ("/heart_disease/", HEART),
        ("/migraine/", {
            "Age": "32", "Gender": "Female", "SleepHours": "6.5",
            "WaterIntake": "5", "SkippedMeals": "Yes", "Caffeine": "2",
            "Stress": "5", "ScreenTime": "8", "PhysicalActivity": "0",
            "Menstruating": "1",
        }),
        ("/health-score/", {
            "Age": "35", "BMI": "22", "ExerciseFrequency": "5",
            "DietQuality": "85", "SleepHours": "8", "SmokingStatus": "0",
            "AlcoholConsumption": "1",
        }),
    ]
    for path, form in cases:
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


def test_sleep_training_range_is_genuinely_narrow():
    """Documents why tier 3 matters, and fails loudly if a retrain changes it.

    The model is trained on people with systolic 110-144 and resting heart rates
    of 60-89. Real users routinely fall outside both.
    """
    profile = load_metadata("sleep")["raw_profile"]
    assert profile["Systolic"]["max"] <= 150
    assert profile["Heart Rate"]["min"] >= 55
