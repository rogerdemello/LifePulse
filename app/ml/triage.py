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
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Concern:
    key: str
    title: str
    blurb: str
    endpoint: str
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
        icon="bi-heart-pulse",
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
        icon="bi-moon-stars",
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
        icon="bi-lightning",
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
        icon="bi-clipboard2-pulse",
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
        icon="bi-calculator",
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
        icon="bi-egg-fried",
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


def route(text, limit=3):
    """Route a description to assessments, using the LLM when it is available.

    Returns ``(matches, method)`` where ``method`` is "model" or "keywords", so
    the page can describe honestly how it reached the answer.

    Three rules this must never break:

    1. **Emergency detection is not in this path.** ``check_emergency`` runs in
       the route before this is called, on deterministic keyword rules, and its
       result cannot be overridden. Whether someone is told to call an ambulance
       must not depend on a third party being reachable.
    2. **The LLM may only choose from the fixed concern set.** Its answer is
       looked up in ``CONCERNS_BY_KEY`` and discarded if it is not a real key,
       so it can route but never invent a destination.
    3. **Any failure falls back to keywords.** Azure being unconfigured, slow,
       rate-limited or wrong is an ordinary condition, not an error page.
    """
    text = (text or "").strip()
    if not text:
        return [], "keywords"

    try:
        matches = _match_with_model(text, limit)
        if matches:
            return matches, "model"
    except Exception as exc:  # never let this break the page
        log.info("falling back to keyword routing: %s", exc)

    return match_concerns(text, limit), "keywords"


def _match_with_model(text, limit):
    """Ask Azure OpenAI which of the fixed concerns this describes.

    The only thing this app ever sends to Azure: the sentence the person typed
    into the "what's bothering you" box. No assessment answers, no results.
    """
    from app.azure_openai import complete_json, is_configured

    if not is_configured():
        return []

    catalogue = "\n".join(f"- {c.key}: {c.title} — {c.blurb}" for c in CONCERNS)
    system = (
        "You route a person's description of a health concern to one or more "
        "screening tools. Reply with JSON: {\"concerns\": [\"key\", ...]} using "
        "at most 3 keys, most relevant first, chosen only from the list given. "
        "Return an empty list if nothing fits. Do not diagnose, do not give "
        "advice, and do not add any other field."
    )
    user = f"Available tools:\n{catalogue}\n\nThe person wrote: {text!r}"

    payload = complete_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=80,
    )

    # Never trust the reply to name a real destination.
    keys, seen = [], set()
    for key in payload.get("concerns") or []:
        concern = CONCERNS_BY_KEY.get(str(key).strip())
        if concern and concern.key not in seen:
            seen.add(concern.key)
            keys.append(concern)

    # The page shows what a match was based on. With a model there are no
    # keywords to show, so it says so rather than inventing some.
    return [(concern, []) for concern in keys[:limit]]


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
