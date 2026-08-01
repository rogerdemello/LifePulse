"""Lifestyle score: a transparent rubric, not a model.

This replaces a RandomForest fit to ``data/synthetic_health_data.csv``. That
model was unsalvageable for three separate reasons:

1. The data was generated, not observed. Every column is Gaussian noise around
   a formula, with no clamping -- 70 rows had *negative* alcohol consumption,
   18 had a diet quality above the stated maximum of 100, and one respondent
   was 1.1 years old. Its R-squared measured how well it recovered someone's
   random number generator.
2. The form's encodings did not match the data's. The diet dropdown offers 1-9
   while training saw 19.9-110.3, so choosing "9 - Excellent" put the user
   *below* the worst diet the model had ever seen. A maximally healthy profile
   scored 68.9 out of 100.
3. Being a model at all bought nothing. There is no ground truth here to
   predict -- "health score" is a construct, not a measurable outcome.

A rubric is better on every axis that matters. It is explainable by
construction, every weight is a stated judgement traceable to public guidance,
and it cannot silently drift from the form. Where the number is uncertain, it
is uncertain for reasons a person can read and disagree with.

Weights are a deliberate editorial choice, documented below. They are not
derived from outcome data, and this file should not pretend otherwise.

Sources:
  Smoking      WHO: tobacco is the leading preventable cause of death.
  Activity     WHO 2020 guidelines: 150-300 min moderate activity per week.
  BMI          WHO classification: healthy range 18.5-24.9.
  Sleep        AASM/SRS consensus: 7-9 hours per night for adults.
  Alcohol      UK Chief Medical Officers: no more than 14 units per week.
  Diet         Self-rated, so weighted lowest -- it is the least objective input.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    """One scored factor and the reasoning behind its score."""

    key: str
    label: str
    weight: int
    score: float          # 0-100 for this factor alone
    answer: str           # what the user said, in words
    verdict: str          # short judgement
    guidance: str         # what the evidence says

    @property
    def earned(self):
        """Points contributed to the overall score."""
        return self.weight * self.score / 100.0

    @property
    def lost(self):
        return self.weight - self.earned


@dataclass(frozen=True)
class LifestyleScore:
    total: float
    components: tuple
    band: str
    colour: str
    interpretation: str

    @property
    def biggest_opportunity(self):
        """The factor where the most points are available, if any."""
        losses = [c for c in self.components if c.lost >= 1]
        return max(losses, key=lambda c: c.lost) if losses else None


# --------------------------------------------------------------------------
# individual factors
#
# Each returns (score 0-100, answer, verdict, guidance).
# --------------------------------------------------------------------------

def _smoking(code):
    """Form: 0 never, 1 former, 2 light (<10/day), 3 moderate (10-20), 4 heavy (>20)."""
    table = {
        0: (100, "never smoked", "Best possible",
            "Not smoking is the single largest modifiable factor in long-term health."),
        1: (80, "former smoker", "Risk falls the longer you stay stopped",
            "Excess cardiovascular risk falls substantially within a few years of quitting."),
        2: (35, "light smoker, under 10 a day", "Still carries substantial risk",
            "There is no safe level of smoking; even a few a day raises cardiovascular risk."),
        3: (15, "moderate smoker, 10-20 a day", "High risk",
            "Quitting at any age improves life expectancy."),
        4: (0, "heavy smoker, over 20 a day", "Highest risk",
            "This is the factor most worth discussing with a doctor. Support makes quitting far more likely to succeed."),
    }
    return table.get(int(code), table[4])


def _activity(days):
    """Form: days per week of exercise, 0-7. WHO asks for 150+ min moderate weekly."""
    days = float(days)
    if days >= 5:
        return 100, f"{days:g} days a week", "Meets or exceeds guidance", \
            "WHO recommends 150-300 minutes of moderate activity weekly."
    if days >= 3:
        return 75, f"{days:g} days a week", "Around the recommended minimum", \
            "Roughly meets the 150-minute target if sessions are 30 minutes or more."
    if days >= 1:
        return 40, f"{days:g} days a week", "Below guidance", \
            "Any activity beats none; the largest gains come from moving from none to some."
    return 0, "no regular exercise", "Sedentary", \
        "Going from nothing to two short sessions a week is the biggest single improvement available here."


def _bmi(value):
    """WHO bands. BMI is crude and fits some bodies badly -- said so in the guidance."""
    bmi = float(value)
    if 18.5 <= bmi < 25:
        return 100, f"{bmi:g}", "Healthy range", \
            "WHO puts the healthy range at 18.5-24.9. BMI is a crude measure and fits athletic or older bodies poorly."
    if 25 <= bmi < 30:
        return 65, f"{bmi:g}", "Overweight range", \
            "Modest, sustained weight loss improves blood pressure and blood sugar."
    if 17 <= bmi < 18.5:
        return 65, f"{bmi:g}", "Underweight range", \
            "Being underweight carries its own risks; worth raising with a doctor."
    if 30 <= bmi < 35:
        return 35, f"{bmi:g}", "Obese range (class I)", \
            "Worth a conversation with a doctor about realistic targets."
    if bmi < 17:
        return 15, f"{bmi:g}", "Severely underweight", \
            "This is worth medical attention rather than self-management."
    return 15, f"{bmi:g}", "Obese range (class II+)", \
        "Worth a conversation with a doctor about realistic targets and support."


def _sleep(hours):
    """AASM/SRS: 7-9 hours for adults. Too much is a signal too, not just too little."""
    h = float(hours)
    if 7 <= h <= 9:
        return 100, f"{h:g} hours", "In the recommended range", \
            "Adults are advised to get 7-9 hours a night."
    if 6 <= h < 7 or 9 < h <= 10:
        return 70, f"{h:g} hours", "Slightly outside the range", \
            "Consistency matters as much as duration."
    if 5 <= h < 6:
        return 40, f"{h:g} hours", "Short sleep", \
            "Regular short sleep is linked to raised cardiovascular and metabolic risk."
    if h > 10:
        return 50, f"{h:g} hours", "Long sleep", \
            "Regularly sleeping over 10 hours can point to an underlying issue worth investigating."
    return 15, f"{h:g} hours", "Very short sleep", \
        "This is worth raising with a doctor, particularly alongside daytime tiredness."


def _alcohol(code):
    """Form: 0 none, 1 rare(1-2), 2 light(3-5), 3 moderate(6-10), 4 heavy(11-15), 5 very heavy(15+).

    UK CMO guidance is no more than 14 units a week. Roughly one drink is one
    to two units, so the boundary falls between the "moderate" and "heavy" bands.
    """
    table = {
        0: (100, "none", "No alcohol-related risk", "No level of drinking is risk-free, so none is the lowest risk."),
        1: (95, "1-2 drinks a week", "Well within guidance", "UK guidance is no more than 14 units weekly."),
        2: (85, "3-5 drinks a week", "Within guidance", "UK guidance is no more than 14 units weekly."),
        3: (60, "6-10 drinks a week", "Approaching the guideline limit", "Spreading drinks across the week and having drink-free days lowers risk."),
        4: (25, "11-15 drinks a week", "At or above the guideline limit", "This is above the 14-unit weekly guideline for most drink sizes."),
        5: (0, "over 15 drinks a week", "Well above the guideline limit", "Worth discussing with a doctor, especially alongside liver or blood pressure concerns."),
    }
    return table.get(int(code), table[5])


def _diet(rating):
    """Form: 1-9 self-rating. Self-reported, so weighted lowest of the six."""
    r = float(rating)
    if r >= 8:
        return 100, f"{r:g} out of 9", "Self-rated as excellent", \
            "Self-assessment is a rough guide; a dietitian can be more specific."
    if r >= 6:
        return 75, f"{r:g} out of 9", "Self-rated as good", \
            "Small consistent changes tend to outlast large short-lived ones."
    if r >= 4:
        return 50, f"{r:g} out of 9", "Self-rated as average", \
            "More vegetables and less ultra-processed food is the change with the broadest evidence."
    return 20, f"{r:g} out of 9", "Self-rated as poor", \
        "This is a large opportunity, and the one most people find easiest to start on."


# Weights total 100. An editorial judgement, ordered by the strength of the
# evidence linking each factor to long-term outcomes, and by how much of it a
# person can actually change.
FACTORS = (
    ("smoking", "Smoking", 25, _smoking, "Smoking_Status"),
    ("activity", "Physical activity", 20, _activity, "Exercise_Frequency"),
    ("bmi", "Body mass index", 15, _bmi, "BMI"),
    ("sleep", "Sleep", 15, _sleep, "Sleep_Hours"),
    ("alcohol", "Alcohol", 15, _alcohol, "Alcohol_Consumption"),
    ("diet", "Diet", 10, _diet, "Diet_Quality"),
)

BANDS = (
    (85, "Strong", "success",
     "Your modifiable lifestyle factors are in good shape across the board."),
    (70, "Good", "success",
     "A solid foundation, with one or two areas worth attention."),
    (55, "Mixed", "primary",
     "Several factors are working for you and several against."),
    (40, "Room to improve", "warning",
     "There are clear opportunities here, and they compound."),
    (0, "Worth a conversation", "danger",
     "Several factors are working against you. A doctor can help you decide where to start."),
)

REQUIRED = tuple(field for *_, field in FACTORS)


def score_lifestyle(values):
    """Score the modifiable factors. ``values`` uses the canonical raw names.

    Age is deliberately absent. This measures what a person can change, and
    scoring someone down for getting older tells them nothing useful.
    """
    missing = [f for f in REQUIRED if f not in values]
    if missing:
        raise ValueError(f"lifestyle score is missing: {missing}")

    components = []
    for key, label, weight, scorer, field in FACTORS:
        score, answer, verdict, guidance = scorer(values[field])
        components.append(Component(
            key=key, label=label, weight=weight, score=float(score),
            answer=answer, verdict=verdict, guidance=guidance,
        ))

    total = sum(c.earned for c in components)
    band, colour, interpretation = next(
        (b, c, i) for threshold, b, c, i in BANDS if total >= threshold
    )
    return LifestyleScore(
        total=round(total, 1),
        components=tuple(components),
        band=band,
        colour=colour,
        interpretation=interpretation,
    )
