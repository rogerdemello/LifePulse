"""End-to-end checks against the real Flask app."""

import pytest

FORMS = {
    "/heart_disease/": {
        "high_bp": "1", "high_chol": "1", "chol_check": "1", "bmi": "34",
        "smoker": "1", "stroke": "0", "diabetes": "1", "phys_activity": "0",
        "fruit": "0", "veggies": "0", "alcohol": "0", "gen_health": "5",
        "ment_health": "15", "phys_health": "20", "diff_walk": "1",
        "sex": "1", "age": "11",
    },
    "/sleep/": {
        "snoring": "3", "gasping": "0", "sleepiness": "2",
        "insomnia_nights": "4", "insomnia_months": "1", "insomnia_impact": "1",
        "sleep_hours": "5.5",
    },
    "/migraine/": {
        "Age": "32", "Gender": "Female", "SleepHours": "4.5",
        "WaterIntake": "1", "SkippedMeals": "Yes", "Caffeine": "5",
        "Stress": "9", "ScreenTime": "12", "PhysicalActivity": "0",
        "Menstruating": "1",
    },
    "/health-score/": {
        "Age": "35", "BMI": "22", "ExerciseFrequency": "5",
        "DietQuality": "85", "SleepHours": "8", "SmokingStatus": "0",
        "AlcoholConsumption": "1",
    },
}


@pytest.mark.parametrize("path", sorted(FORMS) + ["/", "/health/", "/nutrition/"])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_healthz_reports_every_model_loaded(client):
    response = client.get("/healthz")
    assert response.status_code == 200, response.get_json()
    assert all(response.get_json()["models"].values())


@pytest.mark.parametrize("path", sorted(FORMS))
def test_submitting_a_form_returns_a_result(client, path):
    response = client.post(path, data=FORMS[path])
    assert response.status_code == 200


@pytest.mark.parametrize("path", sorted(FORMS))
def test_missing_fields_are_reported_not_crashed(client, path):
    """Blank fields should name what's missing, not 500 or leak a traceback."""
    partial = dict(list(FORMS[path].items())[:2])
    response = client.post(path, data=partial)
    body = response.get_data(as_text=True)
    assert response.status_code == 400
    assert "Missing:" in body
    assert "Traceback" not in body


@pytest.mark.parametrize("path", sorted(FORMS))
def test_garbage_input_does_not_leak_exception_text(client, path):
    """Routes used to `return f"Error: {e}"` straight to the browser."""
    payload = dict(FORMS[path])
    payload[next(iter(payload))] = "<script>alert(1)</script>"
    response = client.post(path, data=payload)
    body = response.get_data(as_text=True)
    assert response.status_code in (200, 400)
    assert "Traceback" not in body
    assert "<script>alert(1)</script>" not in body


def test_sleep_can_return_a_reassuring_verdict(client):
    """The old two-class model could not tell anyone they were fine."""
    healthy = {
        "snoring": "0", "gasping": "0", "sleepiness": "0",
        "insomnia_nights": "0", "insomnia_months": "0", "insomnia_impact": "0",
        "sleep_hours": "8",
    }
    body = client.post("/sleep/", data=healthy).get_data(as_text=True)
    assert "least likely to have sleep apnea" in body
    assert "No trouble sleeping reported" in body


def test_sleep_accepts_any_blood_pressure(client):
    """The old encoder knew 8 literal readings while the form offered 14.

    Everything in its "High" group -- 140/90, 142/92, i.e. exactly the readings
    that matter -- raised on submit. Blood pressure is optional here now and
    any value is accepted, though a stage-2 reading is still flagged first.
    """
    payload = dict(FORMS["/sleep/"])
    payload.update(systolic="163", diastolic="104")

    flagged = client.post("/sleep/", data=payload)
    assert flagged.status_code == 200
    assert "show my results anyway" in flagged.get_data(as_text=True)

    proceeded = client.post("/sleep/", data={**payload, "acknowledged": "1"})
    assert proceeded.status_code == 200


def test_heart_risk_ordering_is_sensible(client):
    """A high-risk profile must score above a low-risk one."""
    import re

    def risk(form):
        body = client.post("/heart_disease/", data=form).get_data(as_text=True)
        return float(re.search(r"([\d.]+)% estimated risk", body).group(1))

    low = dict(FORMS["/heart_disease/"])
    low.update(
        high_bp="0", high_chol="0", bmi="22", smoker="0", diabetes="0",
        phys_activity="1", fruit="1", veggies="1", gen_health="1",
        ment_health="0", phys_health="0", diff_walk="0", age="3",
    )
    assert risk(FORMS["/heart_disease/"]) > risk(low)
