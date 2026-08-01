"""Input safety: what the app refuses to answer, and what it must interrupt for.

LifePulse is meant to be somewhere you'd check before deciding whether to see a
doctor. That only works if it knows the limits of what it knows. Before this
module existed, submitting a blood pressure of 190/125 with a resting heart rate
of 0 returned a calm "No Sleep Disorder" -- a reassuring answer to an input that
is either a typo or a medical emergency, and reassuring in both readings.

Three tiers, in order of precedence:

1. IMPOSSIBLE  -- outside human physiology. Almost certainly a typo. Refuse to
                  predict and say which field. Guessing here is worse than
                  declining, because the user may act on the answer.
2. RED FLAG    -- physiologically possible and medically urgent. Interrupt with
                  "contact a doctor", before any model runs. The user can
                  continue to their results, but not without seeing this.
3. OUT OF RANGE-- possible, not urgent, but outside what the model was trained
                  on. Answer, and say plainly that the answer is unreliable.

Tier 3's bounds come from the data (``raw_profile`` in each model's
metadata.json, written by ml_model/train_all.py) so they track retraining.
Tiers 1 and 2 are clinical facts and are written down here, with sources.

The thresholds below follow widely published guidance. They have NOT been
reviewed by a clinician, and should be before this is promoted as a
pre-appointment tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ImpossibleValue(ValueError):
    """A submitted value falls outside human physiology."""

    def __init__(self, field_name, value, low, high, unit=""):
        self.field_name = field_name
        self.value = value
        suffix = f" {unit}" if unit else ""
        super().__init__(
            f"{field_name}: {value:g}{suffix} is outside the possible range "
            f"({low:g}-{high:g}{suffix}). Please check what you entered."
        )


# --------------------------------------------------------------------------
# Tier 1 -- physiological limits
#
# Deliberately generous: these are "no living person has this" bounds, not
# "unusual" bounds. Anything merely unusual is handled by tier 2 or tier 3.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Limit:
    low: float
    high: float
    unit: str = ""
    label: str = ""


PHYSIOLOGICAL_LIMITS = {
    # Recorded human BP extremes sit well inside these.
    "Systolic": Limit(60, 260, "mmHg", "Systolic blood pressure"),
    "Diastolic": Limit(30, 160, "mmHg", "Diastolic blood pressure"),
    # Resting heart rate. Elite endurance athletes reach the high 20s.
    "Heart Rate": Limit(25, 220, "bpm", "Resting heart rate"),
    # BMI: the lowest survivable is around 10; the heaviest recorded ~100.
    "BMI": Limit(10, 100, "", "BMI"),
    "Sleep Duration": Limit(0, 24, "hours", "Sleep duration"),
    "Sleep Hours": Limit(0, 24, "hours", "Sleep hours"),
    "Age": Limit(0, 120, "years", "Age"),
    "Daily Steps": Limit(0, 100_000, "steps", "Daily steps"),
    "Water Intake": Limit(0, 20, "glasses", "Water intake"),
    "Screen Time": Limit(0, 24, "hours", "Screen time"),
    # BRFSS "days unwell in the last 30".
    "MentHlth": Limit(0, 30, "days", "Days of poor mental health"),
    "PhysHlth": Limit(0, 30, "days", "Days of poor physical health"),
    "GenHlth": Limit(1, 5, "", "General health rating"),
}

# Every form collects age in years, so the shared limit above is the right one
# for all of them. Heart converts years to the BRFSS 5-year bucket *after* this
# check, via the ``transform`` hook in app/routes/support.py -- the check reasons
# about what a person typed, not about the model's internal encoding.
MODEL_LIMIT_OVERRIDES = {}


def limits_for(model_name):
    limits = dict(PHYSIOLOGICAL_LIMITS)
    limits.update(MODEL_LIMIT_OVERRIDES.get(model_name, {}))
    return limits


def check_possible(model_name, values):
    """Raise ``ImpossibleValue`` for the first field outside human physiology."""
    limits = limits_for(model_name)
    for name, limit in limits.items():
        if name not in values:
            continue
        try:
            number = float(values[name])
        except (TypeError, ValueError):
            continue  # the feature builder reports non-numeric input
        if not (limit.low <= number <= limit.high):
            raise ImpossibleValue(
                limit.label or name, number, limit.low, limit.high, limit.unit
            )


# --------------------------------------------------------------------------
# Tier 2 -- red flags
#
# Sources:
#   Blood pressure stages: American Heart Association, "Understanding Blood
#     Pressure Readings" -- hypertensive crisis >180 systolic and/or >120
#     diastolic; stage 2 hypertension >=140/90.
#   Bradycardia <60 bpm and tachycardia >100 bpm are the standard resting
#     definitions; <40 and >120 are used here as the thresholds worth
#     interrupting a wellness tool for.
#   BMI <16 is WHO "severe thinness"; >=40 is WHO obesity class III.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RedFlag:
    key: str
    urgency: str          # "emergency" | "soon"
    title: str
    detail: str
    fields: tuple = field(default=())


def _num(values, name):
    try:
        return float(values[name])
    except (KeyError, TypeError, ValueError):
        return None


def check_red_flags(values):
    """Return red flags for the submitted values, most urgent first.

    Runs before any model. A hypertensive crisis is not something to mention
    underneath a sleep-hygiene tip.
    """
    flags = []
    systolic = _num(values, "Systolic")
    diastolic = _num(values, "Diastolic")
    heart_rate = _num(values, "Heart Rate")
    bmi = _num(values, "BMI")

    if systolic is not None and diastolic is not None:
        if systolic > 180 or diastolic > 120:
            flags.append(RedFlag(
                key="bp_crisis",
                urgency="emergency",
                title="Your blood pressure reading needs urgent attention",
                detail=(
                    f"You entered {systolic:.0f}/{diastolic:.0f} mmHg. Readings above "
                    "180/120 are what the American Heart Association calls a "
                    "hypertensive crisis. If this reading is accurate, contact a "
                    "doctor or urgent care now — this matters more than anything "
                    "else on this page. If you have chest pain, breathlessness, "
                    "weakness or trouble speaking, call emergency services."
                ),
                fields=("Systolic", "Diastolic"),
            ))
        elif systolic >= 140 or diastolic >= 90:
            flags.append(RedFlag(
                key="bp_stage2",
                urgency="soon",
                title="Your blood pressure is in the high range",
                detail=(
                    f"You entered {systolic:.0f}/{diastolic:.0f} mmHg, which falls in "
                    "stage 2 hypertension (140/90 or above). This is worth raising "
                    "with a doctor even if you feel well. A single reading is not a "
                    "diagnosis — blood pressure varies through the day."
                ),
                fields=("Systolic", "Diastolic"),
            ))

    if heart_rate is not None:
        if heart_rate < 40:
            flags.append(RedFlag(
                key="bradycardia",
                urgency="emergency",
                title="Your resting heart rate is unusually low",
                detail=(
                    f"You entered {heart_rate:.0f} bpm. Below 40 at rest is worth "
                    "medical attention promptly, unless you are a trained endurance "
                    "athlete and know this is normal for you. Seek care sooner if "
                    "you feel faint, dizzy or short of breath."
                ),
                fields=("Heart Rate",),
            ))
        elif heart_rate > 120:
            flags.append(RedFlag(
                key="tachycardia",
                urgency="soon",
                title="Your resting heart rate is unusually high",
                detail=(
                    f"You entered {heart_rate:.0f} bpm. A sustained resting rate above "
                    "120 is worth getting checked, particularly alongside chest pain, "
                    "breathlessness or palpitations."
                ),
                fields=("Heart Rate",),
            ))

    if bmi is not None:
        if bmi < 16:
            flags.append(RedFlag(
                key="bmi_low",
                urgency="soon",
                title="Your BMI is in the severely underweight range",
                detail=(
                    f"A BMI of {bmi:.1f} is below 16, which the WHO classifies as "
                    "severe thinness. This is worth discussing with a doctor. BMI is "
                    "a crude measure and does not fit everyone."
                ),
                fields=("BMI",),
            ))
        elif bmi >= 40:
            flags.append(RedFlag(
                key="bmi_high",
                urgency="soon",
                title="Your BMI is in the highest obesity category",
                detail=(
                    f"A BMI of {bmi:.1f} is 40 or above (WHO class III). This carries "
                    "raised risk for several conditions and is worth a conversation "
                    "with a doctor. BMI is a crude measure and does not fit everyone."
                ),
                fields=("BMI",),
            ))

    flags.sort(key=lambda f: 0 if f.urgency == "emergency" else 1)
    return flags


# --------------------------------------------------------------------------
# Tier 3 -- outside what the model was trained on
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Caveat:
    field_name: str
    value: str
    trained_range: str
    message: str


def check_training_range(values, raw_profile):
    """Flag numeric inputs the model has no training evidence for.

    This matters more than it looks. The sleep dataset contains no systolic
    reading above 144 and no resting heart rate outside 60-89, so for anyone
    hypertensive the model is extrapolating with nothing to extrapolate from --
    and it will still return a confident-looking answer.

    Numeric fields only, deliberately. Categorical answers arrive in whatever
    encoding the form uses -- the migraine form posts ``Menstruating=1`` where
    the training data reads "Yes" -- and comparing those strings raises a caveat
    on a perfectly valid answer. The feature builders already reject categories
    they cannot interpret (``_categorical`` in app/ml/features.py), and every
    category they do accept is present in training, so nothing is lost. A
    caveat that cries wolf is worse than no caveat at all: it trains people to
    scroll past the one that matters.
    """
    caveats = []
    for name, profile in (raw_profile or {}).items():
        if name not in values or profile.get("kind") != "numeric":
            continue
        try:
            number = float(values[name])
        except (TypeError, ValueError):
            continue

        low, high = profile["min"], profile["max"]
        if low <= number <= high:
            continue

        caveats.append(Caveat(
            field_name=name,
            value=f"{number:g}",
            trained_range=f"{low:g}–{high:g}",
            message=(
                f"Your {name.lower()} ({number:g}) is outside the range this model "
                f"was trained on ({low:g}–{high:g}). It has not seen anyone like "
                f"you on this measure, so treat the result below as unreliable."
            ),
        ))
    return caveats
