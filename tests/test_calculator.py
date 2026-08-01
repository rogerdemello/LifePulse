"""The health calculator must answer from what the user entered.

It used to hardcode `water_intake_l=2` and `smokes_per_day=0` while the form
asked for neither, so every user in the 57-71 kg band was told "Moderately
Hydrated - increase your water intake" about a number they never gave, and no
smoker was ever warned about smoking. Separately, the activity multiplier was
looked up with `.get(level, 1.2)` against a table keyed "very active" while the
form posted "very_active", so choosing Very Active silently returned the
sedentary calorie figure.

Both are the same failure as the original ML bug: a missing input quietly
replaced by a default, and presented as a personalised finding.
"""

import re

import pytest

from app.utils.calculator import ACTIVITY_MULTIPLIERS, daily_calorie_needs

FORM = {
    "gender": "Male", "age": "40", "activity": "moderate", "height": "175",
    "weight": "70", "waist": "85", "hip": "95", "systolic": "120",
    "diastolic": "80", "water_intake": "2.5", "smokes_per_day": "0",
}


def _body(client, **overrides):
    response = client.post("/health/result", data={**FORM, **overrides})
    assert response.status_code == 200, response.status_code
    return response.get_data(as_text=True)


# --------------------------------------------------------------------------
# inputs must come from the form
# --------------------------------------------------------------------------

def test_hydration_advice_follows_the_water_entered(client):
    assert "Low Hydration" in _body(client, water_intake="0.3")
    assert "Low Hydration" not in _body(client, water_intake="3.5")


def test_smokers_are_warned_and_non_smokers_are_not(client):
    assert "Smoking Impact" in _body(client, smokes_per_day="20")
    assert "Smoking Impact" not in _body(client, smokes_per_day="0")


@pytest.mark.parametrize("field", ["water_intake", "smokes_per_day"])
def test_the_new_fields_are_required_not_defaulted(client, field):
    """A blank field must be reported, never silently filled in."""
    payload = {k: v for k, v in FORM.items() if k != field}
    response = client.post("/health/result", data=payload)
    assert response.status_code == 400
    assert field in response.get_data(as_text=True)


# --------------------------------------------------------------------------
# activity multiplier
# --------------------------------------------------------------------------

def test_activity_level_actually_changes_calorie_needs(client):
    def calories(level):
        numbers = re.findall(r">\s*(\d{4}\.?\d*)\s*<", _body(client, activity=level))
        return [float(n) for n in numbers]

    sedentary, moderate, very_active = (
        calories("sedentary"), calories("moderate"), calories("very_active")
    )
    # BMR is identical across all three; the calorie figure must not be.
    assert sedentary[0] == moderate[0] == very_active[0]
    assert sedentary[1] < moderate[1] < very_active[1]


def test_form_value_very_active_maps_to_the_very_active_multiplier():
    """The exact mismatch that returned sedentary calories for active users."""
    assert daily_calorie_needs(2000, "very_active") == 2000 * ACTIVITY_MULTIPLIERS["very active"]
    assert daily_calorie_needs(2000, "very_active") != 2000 * ACTIVITY_MULTIPLIERS["sedentary"]


def test_unknown_activity_level_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown activity level"):
        daily_calorie_needs(2000, "occasionally")


# --------------------------------------------------------------------------
# disclaimer and privacy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/", "/privacy", "/health/", "/sleep/", "/heart_disease/",
    "/migraine/", "/health-score/", "/nutrition/",
])
def test_every_page_carries_the_same_disclaimer(client, path):
    body = client.get(path).get_data(as_text=True)
    assert "screening tool, not a diagnosis" in body
    assert "/privacy" in body


def test_privacy_page_is_specific_about_the_exceptions(client):
    """A privacy promise that omits the outbound calls is not a promise."""
    body = client.get("/privacy").get_data(as_text=True)
    assert "USDA" in body          # the nutrition lookup does leave the server
    assert "CDN" in body or "cdn" in body
    assert "No database" in body or "no database" in body
