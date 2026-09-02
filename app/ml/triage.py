"""Routing someone from what they've noticed to the right assessment.

Until now you had to already know you wanted the sleep screening. That is the
wrong way round: people arrive with a symptom, not a category. "I keep waking
up with headaches" is what someone types; deciding that maps to the migraine
page is the app's job, not theirs.

Two deliberate design choices:

**Emergency symptoms are checked first and never routed.** Someone typing
"chest pain" must not be handed a questionnaire. That check runs before any
matching and its result cannot be dismissed by a better keyword match.

**Matching is keyword-based and always shows its working.** No attempt is made
to understand a sentence. Matched terms are shown back and the full list of
assessments stays visible, so a wrong guess costs a glance rather than sending
someone down the wrong path believing they were understood.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import namedtuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Concern:
    key: str
    title: str
    blurb: str
    endpoint: str
    # A Bootstrap Icons name with no prefix -- "heart-pulse", not
    # "bi-heart-pulse". The templates build a sprite reference from it as
    # ``#i-{{ concern.icon }}``, and tools/build_icon_sprite.py reads these
    # names when deciding which glyphs to embed, so a name that is not a real
    # icon becomes a blank space rather than an error.
    icon: str
    keywords: tuple = field(default=())


@dataclass(frozen=True)
class EmergencySignal:
    key: str
    title: str
    detail: str
    keywords: tuple


# --------------------------------------------------------------------------
# Emergency symptoms.
#
# These are the classic "call an ambulance" presentations. The wording follows
# widely published public guidance (NHS "call 999" and AHA heart-attack and
# stroke warning signs). This is not triage and makes no attempt to be
# exhaustive -- it is a stop sign in front of a wellness questionnaire.
#
# Like app/ml/safety.py, this has NOT been reviewed by a clinician.
# --------------------------------------------------------------------------

EMERGENCY_SIGNALS = (
    EmergencySignal(
        key="cardiac",
        title="Chest pain or pressure",
        detail=(
            "Chest pain, tightness or pressure — especially with breathlessness, "
            "sweating, nausea, or pain spreading to the arm, neck or jaw — can be "
            "a heart attack. This needs emergency help now, not a questionnaire."
        ),
        keywords=(
            "chest pain", "chest pains", "chest tight", "chest tightness",
            "chest pressure", "crushing chest", "heart attack", "angina",
            "pain in my chest", "pain in chest",
        ),
    ),
    EmergencySignal(
        key="stroke",
        title="Signs of a stroke",
        detail=(
            "Sudden weakness or numbness on one side, a drooping face, trouble "
            "speaking or understanding, or sudden loss of vision are signs of a "
            "stroke. Treatment is time-critical — call emergency services now."
        ),
        keywords=(
            "stroke", "face drooping", "droopy face", "slurred speech",
            "cant speak", "can't speak", "trouble speaking", "one side numb",
            "numb on one side", "weakness on one side", "sudden vision loss",
            "arm weakness",
        ),
    ),
    EmergencySignal(
        key="breathing",
        title="Severe difficulty breathing",
        detail=(
            "Struggling to breathe, or breathlessness that came on suddenly or "
            "stops you speaking in full sentences, needs emergency assessment now."
        ),
        keywords=(
            "cant breathe", "can't breathe", "cannot breathe", "struggling to breathe",
            "gasping for air", "choking", "severe shortness of breath",
        ),
    ),
    EmergencySignal(
        key="neuro",
        title="Sudden severe headache or collapse",
        detail=(
            "A headache that arrives suddenly and is the worst you have ever had, "
            "a first-ever seizure, fainting, or confusion that came on quickly all "
            "need urgent medical assessment."
        ),
        keywords=(
            "worst headache", "thunderclap headache", "sudden severe headache",
            "had a seizure", "having a seizure", "passed out", "fainted",
            "blacked out", "lost consciousness", "sudden confusion",
        ),
    ),
    EmergencySignal(
        key="selfharm",
        title="Thoughts of harming yourself",
        detail=(
            "If you are thinking about harming yourself or ending your life, please "
            "talk to someone now. Contact your local emergency number or a crisis "
            "line — in the US call or text 988, in the UK call 111 or Samaritans on "
            "116 123. You deserve support from a person, not a form."
        ),
        keywords=(
            "kill myself", "killing myself", "suicidal", "suicide",
            "end my life", "ending my life", "take my own life", "end it all",
            "self harm", "selfharm", "harm myself", "hurt myself",
            "want to die", "dont want to live", "no reason to live",
            "better off dead",
        ),
    ),
)


CONCERNS = (
    Concern(
        key="heart",
        title="My heart, blood pressure or cholesterol",
        blurb="Worried about cardiovascular risk, or you've been told your blood "
              "pressure or cholesterol is high.",
        endpoint="heart_disease.predict_heart_disease",
        icon="heart-pulse",
        keywords=(
            "heart", "cardiac", "cardiovascular", "blood pressure", "bp",
            "hypertension", "cholesterol", "palpitations", "circulation",
            "family history of heart", "stroke risk", "diabetes",
        ),
    ),
    Concern(
        key="sleep",
        title="My sleep, snoring or daytime tiredness",
        blurb="Snoring, waking unrested, trouble falling or staying asleep, or "
              "feeling sleepy all day.",
        endpoint="sleep.predict_sleep",
        icon="moon-stars",
        keywords=(
            "sleep", "sleeping", "snore", "snoring", "insomnia", "tired",
            "tiredness", "exhausted", "fatigue", "cant sleep", "can't sleep",
            "waking up", "wake up", "apnea", "apnoea", "drowsy", "sleepy",
            "restless", "nightmares", "wake unrested",
        ),
    ),
    Concern(
        key="migraine",
        title="Headaches or migraines",
        blurb="Recurring headaches, and what might be triggering them.",
        endpoint="migraine.predict_migraine",
        icon="lightning",
        keywords=(
            "headache", "migraine", "head pain", "head hurts", "head hurting",
            "sore head", "pounding head", "throbbing head", "aura",
            "light sensitivity", "photophobia", "temple pain",
        ),
    ),
    Concern(
        key="lifestyle",
        title="My general health and habits",
        blurb="Where your diet, exercise, sleep, smoking and drinking are helping "
              "or hurting, and which to change first.",
        endpoint="health_score.predict_health_score",
        icon="clipboard2-pulse",
        keywords=(
            "lifestyle", "general health", "overall health", "healthy",
            "unhealthy", "exercise", "fitness", "diet", "eating", "smoking",
            "smoke", "alcohol", "drinking", "weight", "obese", "overweight",
            "out of shape", "get healthier", "improve my health",
        ),
    ),
    Concern(
        key="calculator",
        title="My BMI, calories or body measurements",
        blurb="Work out BMI, daily calorie needs, waist-hip ratio and blood "
              "pressure category.",
        endpoint="calculator.show_health_form",
        icon="calculator",
        keywords=(
            "bmi", "body mass", "calories", "calorie", "bmr", "metabolism",
            "waist", "hip ratio", "how much should i weigh", "ideal weight",
            "measurements",
        ),
    ),
    Concern(
        key="nutrition",
        title="What's in the food I eat",
        blurb="Look up any food's nutrients, and what its numbers mean.",
        endpoint="nutrition.nutrition_lookup",
        icon="egg-fried",
        keywords=(
            "food", "nutrition", "nutrients", "calories in", "sugar", "salt",
            "sodium", "fat", "protein", "fibre", "fiber", "vitamin", "vitamins",
            "mineral", "what should i eat", "ingredients",
        ),
    ),
)

CONCERNS_BY_KEY = {c.key: c for c in CONCERNS}


def _normalise(text):
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Punctuation goes so "can't sleep" and "cant sleep" both match, which is the
    difference between routing someone and shrugging at them.
    """
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Words too common to carry meaning. Dropped before matching so "end my life"
# matches "thinking about ending my life", where the words are not adjacent.
_STOPWORDS = frozenset(
    "a an and the my me i im is are was were be been of to in on at it its "
    "for with about have has had do does did keep keeps got get getting feel "
    "feeling been very really quite so much lot lots".split()
)


def _stem(word):
    """Crudest possible stemmer: enough for 'ending' to reach 'end'."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _stems(text):
    return [_stem(w) for w in _normalise(text).split() if w not in _STOPWORDS]


def _related(a, b):
    """True when two stems are the same word in different clothes.

    Prefix matching in one direction only, with a floor of three characters, so
    'snor' reaches 'snore' and 'hurt' reaches 'hurting' without 'car' reaching
    'cardiac'.
    """
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 3 and longer.startswith(shorter) and len(longer) - len(shorter) <= 3


def _keyword_matches(keyword, haystack_stems):
    """Every word of ``keyword`` must appear, in any order."""
    needed = _stems(keyword)
    if not needed:
        return False
    return all(any(_related(n, h) for h in haystack_stems) for n in needed)


def check_emergency(text):
    """Emergency signals matched in ``text``. Checked before anything else.

    Every emergency keyword is a multi-word phrase on purpose. A single generic
    word invites the wrong kind of mistake -- "fit" for seizure would fire on
    "I want to improve my fitness", and telling someone to call an ambulance
    about their gym plans destroys the credibility of every warning here.
    """
    stems = _stems(text)
    if not stems:
        return []
    return [
        signal for signal in EMERGENCY_SIGNALS
        if any(_keyword_matches(k, stems) for k in signal.keywords)
    ]


# `route(text)` used to live here: a single-shot version of `converse` that
# took one string and returned (matches, method). `converse` with one user turn
# does exactly that and nothing else called it, so keeping both would have left
# two routing paths free to drift apart -- which is the failure this codebase
# keeps finding elsewhere. Its tests moved to `converse` rather than being
# deleted with it.


# How many clarifying questions the agent may ask before it has to commit.
#
# Enforced here, not in the prompt. A model told "ask at most two questions"
# will usually comply and occasionally will not, and the failure mode is a
# person answering questions forever on a page that never gives them anything.
MAX_QUESTIONS = 2

# Bounds on the transcript, which arrives from hidden form fields and is
# therefore whatever the browser chose to send back. Nothing is stored, so
# tampering only affects the sender -- but it is still assembled into a
# request to Azure, so it gets a ceiling.
MAX_TURNS = 8
MAX_TURN_CHARS = 500

Outcome = namedtuple("Outcome", "action question matches method")


def converse(turns, limit=3):
    """Decide whether to ask one more question or route, given the exchange.

    ``turns`` is a list of ``{"role": "user"|"agent", "text": str}`` in order.
    Returns an ``Outcome`` whose ``action`` is "ask" or "route".

    The rules from ``route`` all still hold, and two more:

    4. **The question budget is enforced here.** After ``MAX_QUESTIONS`` the
       agent must route, whatever it would prefer to do, so the conversation
       cannot continue indefinitely.
    5. **A reply that is neither a real question nor a real destination is a
       failure**, and failures fall back to keywords over everything said so
       far -- not to an error, and not to an empty page.

    Emergency detection is deliberately not here. It runs in the route, on
    every turn, over everything the person has typed, before this is called.
    """
    said = " ".join(t["text"] for t in turns if t["role"] == "user").strip()
    if not said:
        return Outcome("route", None, [], "keywords")

    asked = sum(1 for t in turns if t["role"] == "agent")

    try:
        decision = _decide_with_model(turns, limit, may_ask=asked < MAX_QUESTIONS)
        if decision is not None:
            return decision
    except Exception as exc:  # never let this break the page
        log.info("falling back to keyword routing: %s", exc)

    return Outcome("route", None, match_concerns(said, limit), "keywords")


def _decide_with_model(turns, limit, may_ask):
    """One agent step. Returns an ``Outcome``, or ``None`` to fall back.

    Only what the person typed is sent, plus the agent's own previous
    questions so it does not repeat itself. No assessment answers, no results,
    no scores -- those never reach this module.
    """
    from app.azure_openai import complete_json, is_configured

    if not is_configured():
        return None

    catalogue = "\n".join(f"- {c.key}: {c.title} — {c.blurb}" for c in CONCERNS)
    system = (
        "You help a person find which of a fixed set of health screening tools "
        "fits what they have noticed. "
        + (
            "Reply with JSON, either "
            "{\"action\": \"ask\", \"question\": \"...\"} for ONE short "
            "clarifying question when their description could fit several "
            "tools, or "
            "{\"action\": \"route\", \"concerns\": [\"key\", ...]} when you "
            "can tell. "
            if may_ask else
            "Reply with JSON {\"action\": \"route\", \"concerns\": [\"key\", "
            "...]}. You may not ask a question. "
        )
        + "Use at most 3 keys, most relevant first, chosen only from the list "
        "given, and an empty list if nothing fits. A question must be one "
        "sentence about their symptoms and must not suggest a cause. "
        "Never diagnose, never give medical advice, never tell someone "
        "whether to seek care."
    )

    lines = [f"Available tools:\n{catalogue}\n"]
    for turn in turns:
        who = "The person wrote" if turn["role"] == "user" else "You asked"
        lines.append(f"{who}: {turn['text']!r}")

    payload = complete_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": "\n".join(lines)}],
        max_tokens=120,
    )

    action = str(payload.get("action") or "").strip().lower()

    if action == "ask" and may_ask:
        question = " ".join(str(payload.get("question") or "").split())
        # A blank or absurd question is a failure, not something to render.
        if 8 <= len(question) <= 200:
            return Outcome("ask", question, [], "model")
        return None

    # A reply carrying `concerns` and no `action` is still a usable answer, and
    # discarding it would drop a good route over a missing label.
    if action == "route" or (not action and payload.get("concerns") is not None):
        matches = _resolve(payload.get("concerns"), limit)
        if matches:
            return Outcome("route", None, matches, "model")

    return None


def _resolve(keys, limit):
    """Look raw model output up in the fixed set, discarding anything else."""
    resolved, seen = [], set()
    for key in keys or []:
        concern = CONCERNS_BY_KEY.get(str(key).strip())
        if concern and concern.key not in seen:
            seen.add(concern.key)
            resolved.append(concern)
    # The page shows what a match was based on. With a model there are no
    # keywords to show, so it says so rather than inventing some.
    return [(concern, []) for concern in resolved[:limit]]


def read_turns(pairs):
    """Rebuild the transcript from hidden form fields, defensively.

    The exchange is not stored anywhere: it rides in hidden inputs and comes
    back with the next post, which is the same trick the red-flag interstitial
    uses to replay a submission without a session. That means every value here
    is attacker-controlled, so this validates shape and bounds rather than
    trusting it. It cannot leak across users -- there is nothing to leak into.
    """
    turns = []
    for raw in list(pairs)[:MAX_TURNS]:
        role, _, text = str(raw).partition(":")
        text = text.strip()[:MAX_TURN_CHARS]
        if role in ("user", "agent") and text:
            turns.append({"role": role, "text": text})
    return turns




def match_concerns(text, limit=3):
    """Rank concerns against free text by keyword.

    Returns ``[(concern, matched_terms)]``, best first, empty when nothing
    matched. Longer keyword matches score higher so "blood pressure" beats a
    stray "pressure", and the matched terms are returned so the page can show
    what it keyed on rather than appearing to have understood a sentence.

    Still the fallback, and still the only path when Azure is not configured.
    """
    stems = _stems(text)
    if not stems:
        return []

    scored = []
    for concern in CONCERNS:
        matched = [k for k in concern.keywords if _keyword_matches(k, stems)]
        if not matched:
            continue
        score = sum(len(_stems(k)) * 10 + len(k) for k in matched)
        scored.append((score, concern, sorted(matched, key=len, reverse=True)))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [(concern, matched) for _, concern, matched in scored[:limit]]
