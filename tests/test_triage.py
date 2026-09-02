"""The symptom-led entry point.

People arrive with a symptom, not a category. Until this existed you had to
already know you wanted the sleep screening.

The tests that matter most here are the emergency ones, in both directions:
someone describing a heart attack must never be handed a questionnaire, and
someone describing their gym plans must never be told to call an ambulance. A
warning that fires on "I want to improve my fitness" destroys the credibility
of every other warning in the app.
"""

import pytest

from app.ml.triage import CONCERNS, check_emergency, match_concerns

# --------------------------------------------------------------------------
# emergencies -- must fire
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I have chest pain", "cardiac"),
    ("crushing chest pressure and my arm hurts", "cardiac"),
    ("I think I'm having a heart attack", "cardiac"),
    ("my face is drooping and I can't speak properly", "stroke"),
    ("sudden weakness on one side", "stroke"),
    ("I can't breathe", "breathing"),
    ("gasping for air", "breathing"),
    ("worst headache of my life, came on suddenly", "neuro"),
    ("I blacked out yesterday", "neuro"),
    ("I keep thinking about ending my life", "selfharm"),
    ("I want to kill myself", "selfharm"),
    ("sometimes I feel better off dead", "selfharm"),
    ("I've been harming myself", "selfharm"),
])
def test_emergency_phrasings_are_caught(text, expected):
    assert expected in [s.key for s in check_emergency(text)], text


def test_inflected_phrasing_still_matches():
    """"ending my life" against a keyword of "end my life" -- and the words
    are not even adjacent in "thinking about ending my life"."""
    assert check_emergency("I keep thinking about ending my life")
    assert check_emergency("thoughts of killing myself")


# --------------------------------------------------------------------------
# emergencies -- must NOT fire
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I want to improve my fitness",
    "trying to get fit and lose weight",
    "I get headaches sometimes",
    "I have a mild headache",
    "my chest feels fine",
    "I snore a lot",
    "worried about my cholesterol",
    "what should I eat",
    "I'm tired all the time",
])
def test_ordinary_descriptions_do_not_trigger_an_emergency(text):
    """A stop sign that fires on gym plans teaches people to ignore stop signs."""
    assert check_emergency(text) == [], text


def test_empty_input_is_not_an_emergency():
    assert check_emergency("") == []
    assert check_emergency("   ") == []


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I snore and I'm tired all day", "sleep"),
    ("I can't sleep at night", "sleep"),
    ("my head keeps hurting", "migraine"),
    ("I get migraines", "migraine"),
    ("worried about my blood pressure", "heart"),
    ("my cholesterol is high", "heart"),
    ("I want to eat better", "lifestyle"),
    ("I feel unhealthy and out of shape", "lifestyle"),
    ("what's my BMI", "calculator"),
])
def test_descriptions_route_to_the_right_assessment(text, expected):
    keys = [c.key for c, _ in match_concerns(text)]
    assert keys and keys[0] == expected, f"{text!r} -> {keys}"


def test_no_match_is_an_honest_empty_result():
    """Guessing at nonsense would be worse than admitting nothing matched."""
    assert match_concerns("purple monkey dishwasher") == []
    assert match_concerns("") == []


def test_matches_report_what_they_keyed_on():
    """The page shows these, so a wrong guess costs a glance not a wrong path."""
    matches = match_concerns("I snore badly")
    assert matches
    _, terms = matches[0]
    assert terms
    assert any("snor" in t for t in terms)


def test_longer_phrases_outrank_stray_words():
    keys = [c.key for c, _ in match_concerns("my blood pressure is high")]
    assert keys[0] == "heart"


def test_every_concern_points_at_a_real_endpoint(app):
    with app.test_request_context():
        from flask import url_for
        for concern in CONCERNS:
            assert url_for(concern.endpoint)


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def test_the_page_lists_everything_without_searching(client):
    """Search is an accelerator, not a gate."""
    import html

    body = html.unescape(client.get("/start").get_data(as_text=True))
    for concern in CONCERNS:
        assert concern.title in body


def test_an_emergency_description_is_not_given_a_questionnaire(client):
    response = client.post("/start", data={"concern": "I have crushing chest pain"})
    body = response.get_data(as_text=True)
    assert "Please get medical help now" in body
    # No routing suggestions alongside it.
    assert "This looks like the one" not in body
    assert "These look closest" not in body


def test_self_harm_gets_a_crisis_line_not_a_form(client):
    body = client.post(
        "/start", data={"concern": "I keep thinking about ending my life"}
    ).get_data(as_text=True)
    assert "988" in body or "Samaritans" in body
    assert "Start this assessment" not in body


def test_a_routed_description_offers_the_assessment(client):
    body = client.post(
        "/start", data={"concern": "I snore and wake up tired"}
    ).get_data(as_text=True)
    assert "My sleep, snoring or daytime tiredness" in body
    assert "Start this assessment" in body
    assert "Matched on:" in body


def test_an_unmatched_description_says_so(client):
    body = client.post(
        "/start", data={"concern": "purple monkey dishwasher"}
    ).get_data(as_text=True)
    assert "Nothing matched" in body
    assert "speak to a doctor" in body


def test_the_homepage_leads_with_it(client):
    """The homepage used to lead with a button reading "Tell us what's
    bothering you" that went to /start, where the actual field lived. It now
    carries the field itself and posts to the same route, which is a stronger
    version of the same claim -- so this asserts the input is there rather than
    matching the wording of a button that no longer exists.
    """
    body = client.get("/").get_data(as_text=True)
    assert 'action="/start"' in body
    assert 'name="concern"' in body
