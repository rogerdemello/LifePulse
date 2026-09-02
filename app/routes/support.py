"""Shared helpers for the prediction routes.

Previously each route ended in ``except Exception as e: return f"Error: {e}"``,
which sent raw exception text to the browser as unstyled HTML and logged
nothing. These helpers give the four routes one consistent way to collect form
input, check it is safe to answer at all, surface a usable message, and record
the real error server-side.
"""

from __future__ import annotations

import functools
from datetime import datetime

from flask import current_app, render_template, request

from app.ml.bundle import ModelNotAvailable
from app.ml.features import FeatureContractError, describe_value, label_for
from app.ml.guidance import questions_for
from app.ml.safety import (
    ImpossibleValue,
    check_possible,
    check_red_flags,
    check_training_range,
)
from app.ml.sleep_risk import SLEEPINESS_LABELS, SNORING_LABELS
from app.observability import current_request_id


class FormError(ValueError):
    """A problem with the submitted form that the user can act on."""


def collect(form, mapping):
    """Pull ``mapping`` (form field -> canonical raw name) out of ``form``.

    Raises ``FormError`` naming every blank field at once rather than failing on
    whichever one happens to be read first.
    """
    values, missing = {}, []
    for field, raw_name in mapping.items():
        value = form.get(field)
        if value is None or str(value).strip() == "":
            missing.append(field)
        else:
            values[raw_name] = str(value).strip()
    if missing:
        raise FormError(
            "Please fill in every field. Missing: " + ", ".join(sorted(missing))
        )
    return values


def collect_and_check(form, mapping, model, transform=None):
    """Collect the form, then decide whether it is safe to answer.

    Returns ``(values, caveats)``. Raises ``ImpossibleValue`` when the input is
    outside human physiology, which the error handler turns into a 400 rather
    than a prediction — guessing from a typo is worse than declining, because
    the user may act on the answer.

    ``caveats`` describe inputs the model has no training evidence for. They are
    rendered alongside the result, not instead of it.

    ``transform`` re-encodes collected values into the units the model was
    trained on. It runs *after* the physiological check (which reasons about
    what a human entered) and *before* the training-range check (which reasons
    about what the model saw). Heart uses it to turn age in years into the
    BRFSS 5-year bucket.
    """
    values = collect(form, mapping)
    check_possible(model.name, values)
    if transform is not None:
        values = transform(values)
    caveats = check_training_range(values, model.metadata.get("raw_profile"))
    return values, caveats


def urgent_interstitial(values, form):
    """Render the red-flag interruption, or return ``None`` if there is none.

    A hypertensive crisis should not appear as a footnote under a sleep-hygiene
    tip, so this runs before the model does. The user can always continue —
    ``acknowledged`` replays the original submission — but not without seeing it.

    Stateless by design: the pending answers ride in hidden fields on the
    continue form rather than in a session, so nothing is stored server-side
    and nothing is written to a cookie.
    """
    if form.get("acknowledged"):
        return None
    flags = check_red_flags(values)
    if not flags:
        return None
    return render_template(
        "urgent.html",
        flags=flags,
        emergency=any(f.urgency == "emergency" for f in flags),
        action=request.path,
        submitted=[(k, v) for k, v in form.items(multi=True)],
    )


def build_summary(title, headline, detail, model_name, raw, factors, caveats, form):
    """Assemble everything a person would want to hand a doctor.

    Returned as a plain dict so the result page can embed it as JSON. The
    browser stores it in localStorage if the user chooses to keep it; it is
    never posted back, never written to a cookie, and never touches the server
    again. That is what makes the "nothing is stored" promise literal rather
    than approximate.
    """
    flags = check_red_flags(raw)
    return {
        "title": title,
        "date": datetime.now().strftime("%d %b %Y"),
        "headline": headline,
        "detail": detail,
        "inputs": [
            {"label": label_for(name), "value": describe_value(name, value, model_name)}
            for name, value in raw.items()
        ],
        "factors": [
            {
                "label": f.label,
                "value": f.value,
                "direction": f.direction,
                "delta": round(f.magnitude, 1),
            }
            for f in factors
        ],
        "caveats": [c.message for c in caveats],
        "flags": [
            {"title": f.title, "detail": f.detail, "urgency": f.urgency}
            for f in flags
        ],
        "questions": questions_for(model_name, headline, factors, caveats, flags),
    }


def build_rubric_summary(result, raw, form):
    """Visit summary for the lifestyle score.

    Shaped like ``build_summary`` so the same localStorage summary page renders
    both, but sourced from the rubric's components rather than a model
    explanation -- there is no model here to explain.
    """
    flags = check_red_flags(raw)
    opportunity = result.biggest_opportunity

    questions = []
    for flag in flags:
        questions.append(f"Should we talk about {flag.title.lower()}?")
    questions.append(
        f"I scored {result.total:.0f} out of 100 on a lifestyle checklist. "
        f"Which of these would make the most difference for someone with my history?"
    )
    if opportunity:
        questions.append(
            f"{opportunity.label} came out as my biggest opportunity "
            f"({opportunity.lost:.0f} of {opportunity.weight} points). "
            f"What support is available for that?"
        )

    return {
        "title": "Lifestyle score",
        "date": datetime.now().strftime("%d %b %Y"),
        "headline": f"{result.total:.0f} out of 100 — {result.band}",
        "detail": (
            f"{result.interpretation} Scored against published guidance rather "
            f"than a model, so every point is traceable to a stated rule."
        ),
        "inputs": [
            {"label": component.label, "value": component.answer}
            for component in result.components
        ],
        "factors": [
            {
                "label": component.label,
                "value": component.answer,
                "direction": "lowered" if component.lost >= 1 else "raised",
                "delta": round(max(component.lost, component.earned), 1),
            }
            for component in sorted(
                result.components, key=lambda c: c.lost, reverse=True
            )
            if component.lost >= 1
        ],
        "caveats": [],
        "flags": [
            {"title": f.title, "detail": f.detail, "urgency": f.urgency}
            for f in flags
        ],
        "questions": questions[:6],
    }


def build_sleep_summary(apnea, insomnia, sleep_hours, vitals, form):
    """Visit summary for the sleep screening.

    Same shape as the model-backed ones so the summary page renders them
    identically, but sourced from observed national rates rather than a
    prediction.
    """
    flags = check_red_flags(vitals)
    questions = [f"Should we talk about {f.title.lower()}?" for f in flags]

    if apnea.witnessed_gasping:
        questions.append(
            "Someone has seen me stop breathing or gasp in my sleep. "
            "Should I be referred for a sleep study?"
        )
    elif apnea.band in ("high", "raised"):
        questions.append(
            f"A screening tool put me in a group where {apnea.percent}% report "
            f"signs of sleep apnea. Is a sleep study worth doing?"
        )

    if insomnia.meets_criteria:
        questions.append(
            "My sleep problems meet the definition of chronic insomnia. "
            "Can I be referred for CBT-I rather than sleeping tablets?"
        )
    if not 7 <= sleep_hours <= 9:
        questions.append(
            f"I sleep about {sleep_hours:g} hours a night. Could that be "
            f"behind symptoms I've noticed?"
        )

    return {
        "title": "Sleep screening",
        "date": datetime.now().strftime("%d %b %Y"),
        "headline": apnea.headline,
        "detail": apnea.comparison,
        "inputs": [
            {"label": "Snoring", "value": SNORING_LABELS[apnea.snoring]},
            {"label": "Daytime sleepiness", "value": SLEEPINESS_LABELS[apnea.sleepiness]},
            {"label": "Witnessed gasping or stopping breathing",
             "value": "yes" if apnea.witnessed_gasping else "no"},
            {"label": "Nights a week with trouble sleeping", "value": str(insomnia.nights)},
            {"label": "Going on 3 months or more",
             "value": "yes" if insomnia.months_3_plus else "no"},
            # The form asks "does it affect your daytime -- mood, concentration,
            # or energy?"; "Affects my daytime" alone reads as a truncation on a
            # page a doctor is meant to read.
            {"label": "Affects mood, concentration or energy",
             "value": "yes" if insomnia.daytime_impact else "no"},
            {"label": "Hours of sleep a night", "value": f"{sleep_hours:g}"},
        ],
        "factors": [],
        "caveats": [],
        "flags": [
            {"title": f.title, "detail": f.detail, "urgency": f.urgency}
            for f in flags
        ],
        "questions": questions[:6],
    }


def error_page(message, status=400, title="Something went wrong"):
    # The reference is shown only on server errors: it is there so a user can
    # quote it, and there is nothing to quote when they simply mistyped.
    return render_template(
        "error.html", message=message, title=title,
        reference=current_request_id() if status >= 500 else None,
    ), status


def unavailable_page(name):
    return error_page(
        f"The {name.replace('_', ' ')} model is not loaded on this server, so "
        f"predictions are unavailable. If you are running LifePulse locally, "
        f"build the models with: python ml_model/train_all.py",
        status=503,
        title="Model unavailable",
    )


def prediction_errors(view):
    """Turn expected failures into readable pages and log the unexpected ones.

    ``FormError``, ``ImpossibleValue`` and ``FeatureContractError`` are the
    user's problem to fix and become a 400. ``ModelNotAvailable`` is the
    server's and becomes a 503. Anything else is a bug: it is logged with a
    traceback and reported as a 500 without leaking the exception text.
    """

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        try:
            return view(*args, **kwargs)
        except FormError as exc:
            return error_page(str(exc), status=400, title="Check your answers")
        except ImpossibleValue as exc:
            current_app.logger.info("rejected impossible input: %s", exc)
            return error_page(str(exc), status=400, title="Please check that value")
        except FeatureContractError as exc:
            current_app.logger.warning("feature contract: %s", exc)
            return error_page(
                "Some of the values submitted could not be interpreted. "
                "Please review the form and try again.",
                status=400,
                title="Check your answers",
            )
        except ModelNotAvailable as exc:
            current_app.logger.error("model unavailable: %s", exc)
            return unavailable_page(getattr(exc, "model", "prediction"))
        except Exception:
            current_app.logger.exception("unhandled error in %s", view.__name__)
            return error_page(
                "An unexpected error occurred while generating your result. "
                "The problem has been logged.",
                status=500,
            )

    return wrapper
