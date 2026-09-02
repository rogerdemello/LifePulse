"""The symptom-led entry point.

People arrive with a symptom, not a category. Until this existed you had to
already know you wanted the sleep screening.

The tests that matter most here are the emergency ones, in both directions:
someone describing a heart attack must never be handed a questionnaire, and
someone describing their gym plans must never be told to call an ambulance. A
warning that fires on "I want to improve my fitness" destroys the credibility
of every other warning in the app.
"""

from pathlib import Path

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


# --------------------------------------------------------------------------
# Measured, not sampled
#
# The tests above are hand-picked examples, which is how this check was
# validated for its whole life: fifteen cases, all of them ones somebody had
# thought of. tests/data/emergency_phrasings.csv is a labelled set built to
# include the ones nobody had -- negations, third-person reports, and ordinary
# complaints that happen to contain an emergency phrase's words.
#
# Against it, the original bag-of-stems matcher scored:
#
#     sensitivity 95.6%    false alarms 46.2%
#
# Nearly half of ordinary complaints fired the stop sign. That is not a
# cosmetic problem: a warning that goes off on "my dad had chest pain, am I at
# risk" is one people learn to click past, and it also blocks that person from
# the assessment they came for.
# --------------------------------------------------------------------------

PHRASINGS = Path(__file__).resolve().parent / "data" / "emergency_phrasings.csv"

# Both floors are deliberately asymmetric. A missed emergency is the failure
# this module exists to prevent, so nothing short of catching all of them
# passes. False alarms are the safe direction, so the bound there is a ceiling
# on drift rather than a target.
REQUIRED_SENSITIVITY = 1.0
MAX_FALSE_ALARM_RATE = 0.15


def _labelled():
    rows = []
    for line in PHRASINGS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("text,"):
            continue
        text, expect, category = line.rsplit(",", 2)
        rows.append((text, expect == "yes", category))
    return rows


def _score():
    caught = missed = alarmed = quiet = 0
    misses, alarms = [], []
    for text, is_emergency, category in _labelled():
        fired = bool(check_emergency(text))
        if is_emergency and fired:
            caught += 1
        elif is_emergency:
            missed += 1
            misses.append((category, text))
        elif fired:
            alarmed += 1
            alarms.append((category, text))
        else:
            quiet += 1
    return caught, missed, alarmed, quiet, misses, alarms


def test_the_labelled_set_covers_both_directions():
    """A set of only true emergencies would measure nothing about false alarms,
    which is where this check was actually failing."""
    rows = _labelled()
    assert len(rows) >= 100
    categories = {category for _, _, category in rows}
    for required in ("negated", "attributed", "incidental", "ordinary"):
        assert required in categories, f"the set has no {required} cases"
    assert sum(1 for _, is_emergency, _ in rows if is_emergency) >= 40


def test_no_labelled_emergency_is_missed():
    """The one that must not regress."""
    caught, missed, _, _, misses, _ = _score()
    sensitivity = caught / (caught + missed)
    assert sensitivity >= REQUIRED_SENSITIVITY, (
        f"sensitivity {sensitivity:.1%}; missed:\n  "
        + "\n  ".join(f"[{c}] {t}" for c, t in misses)
    )


def test_false_alarms_stay_within_bounds():
    _, _, alarmed, quiet, _, alarms = _score()
    rate = alarmed / (alarmed + quiet)
    assert rate <= MAX_FALSE_ALARM_RATE, (
        f"false alarm rate {rate:.1%}; fired on:\n  "
        + "\n  ".join(f"[{c}] {t}" for c, t in alarms)
    )


def test_negation_and_attribution_are_handled():
    """The two categories the suppression rules were written for.

    Everything still firing is `incidental` -- an emergency phrase's words
    scattered through an unrelated sentence, like "back pain and a chest
    infection". Those are left alone on purpose: the obvious fix is to require
    the words to sit close together, and that breaks the case this module
    exists for. See _emergency_span.
    """
    _, _, _, _, _, alarms = _score()
    by_category = {}
    for category, text in alarms:
        by_category.setdefault(category, []).append(text)

    assert len(by_category.get("attributed", [])) == 0, (
        "somebody else's symptom still stops the questionnaire: "
        f"{by_category['attributed']}"
    )
    # One survives: "no history of stroke or heart attack", where the cue sits
    # five words from the match, past the window. Widening the window to reach
    # it starts suppressing real emergencies, which is the worse trade.
    assert len(by_category.get("negated", [])) <= 1, by_category.get("negated")


def test_a_denied_symptom_does_not_fire():
    assert not check_emergency("no chest pain but I get breathless on stairs")
    assert not check_emergency("I don't have chest pain")
    assert not check_emergency("I have never fainted in my life")


def test_someone_elses_symptom_does_not_fire():
    """These people are asking for the heart assessment, and firing the stop
    sign is what stands between them and it."""
    assert not check_emergency("my dad had chest pain last year am I at risk")
    assert not check_emergency("family history of heart attack on my dad's side")
    assert not check_emergency("my mother had a stroke should I be checked")


def test_a_negation_cannot_suppress_a_symptom_it_does_not_govern():
    """Both of these carry a negation cue *after* the symptom, negating
    something else entirely. Suppressing on them cost five real emergencies
    when the window looked in both directions."""
    assert check_emergency("chest pressure that won't go away")
    assert check_emergency("everyone would be better off dead without me")


def test_the_keyword_phrase_may_contain_its_own_negation():
    """"I don't want to live" must not negate the keyword "dont want to live"."""
    assert check_emergency("I don't want to live anymore")
    assert check_emergency("I have no reason to live")


def test_a_contraction_is_one_word():
    """_normalise deletes apostrophes rather than splitting on them. While it
    split, "don't" became "don" + "t" and no negation cue the matcher looked
    for ever appeared in real typed text."""
    from app.ml.triage import _tokens

    assert _tokens("I don't have chest pain") == [
        "i", "dont", "have", "chest", "pain"
    ]
    assert check_emergency("I can't breathe")


def test_a_negation_cue_never_stands_in_for_an_ordinary_word():
    """"can" is a three-letter prefix of "cant", so prefix matching made
    "I can breathe fine" match the keyword "cant breathe"."""
    assert not check_emergency("I can breathe fine it's the snoring that bothers me")
    assert check_emergency("I can't breathe properly")
