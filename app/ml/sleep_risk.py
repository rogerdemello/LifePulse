"""Sleep screening from real national data, without a model.

The previous sleep model was trained on a small file whose respondents all had
a systolic blood pressure between 110 and 144 and a resting pulse between 60
and 89. Anyone hypertensive fell outside everything it had seen, so the app
refused to trust its own answer for exactly the people most likely to have
sleep apnea. It also had only two classes and could not tell anyone they were
fine.

Retraining on NHANES 2017-2018 (a real, public-domain US national survey with
*measured* blood pressure and pulse, 4,417 adults, systolic 72-224) fixed the
range problem and then showed something more useful: no model was warranted.

    snoring alone                         ROC-AUC 0.775
    snoring + daytime sleepiness          ROC-AUC 0.791
    ... plus age, sex, BMI, BP and pulse  ROC-AUC 0.741   <- worse
    unfitted rule 2*snoring + sleepiness  ROC-AUC 0.791   <- matches the model

Two questions carry the signal; a fitted model over nine features does no
better than adding them up, and adding vitals actively hurts. So this reports
the observed rate for the answers given, straight from the survey. There is no
training range to fall outside, nothing to extrapolate, and every number can be
checked against the published NHANES files.

Insomnia is handled separately and deliberately not predicted. On the same
data, predicting it from body measurements reaches ROC-AUC 0.616 -- barely
above chance, which is unsurprising: insomnia is defined by its symptoms, not
inferred from someone's blood pressure. Those symptoms are asked about directly
and checked against the standard criteria instead.

Rebuild the table with: python ml_model/fetch_nhanes.py
"""

from __future__ import annotations

from dataclasses import dataclass

# Share of NHANES 2017-18 adults reporting snorting, gasping or stopping
# breathing during sleep at least occasionally, by their own answers to the two
# screening questions. Every cell holds at least 199 respondents.
#
#   snoring    0 never  1 rarely  2 occasionally  3 frequently
#   sleepiness 0 rarely 1 sometimes  2 often/always
#
# (rate, respondents in that cell)
NHANES_APNEA_RATES = {
    (0, 0): (0.01, 574), (0, 1): (0.02, 359), (0, 2): (0.03, 265),
    (1, 0): (0.02, 441), (1, 1): (0.02, 381), (1, 2): (0.05, 253),
    (2, 0): (0.10, 333), (2, 1): (0.12, 330), (2, 2): (0.17, 199),
    (3, 0): (0.19, 436), (3, 1): (0.27, 428), (3, 2): (0.37, 418),
}

NHANES_OVERALL_RATE = 0.115
NHANES_SAMPLE = 4417
NHANES_CYCLE = "NHANES 2017-2018"

SNORING_LABELS = {
    0: "never", 1: "rarely (1-2 nights a week)",
    2: "occasionally (3-4 nights a week)", 3: "frequently (5 or more nights a week)",
}

SLEEPINESS_LABELS = {
    0: "rarely", 1: "sometimes (2-4 times a month)",
    2: "often (5 or more times a month)",
}


@dataclass(frozen=True)
class ApneaRisk:
    rate: float                # observed share in this group
    sample: int                # respondents in this cell
    snoring: int
    sleepiness: int
    band: str                  # "low" | "raised" | "high"
    headline: str
    comparison: str
    witnessed_gasping: bool

    @property
    def percent(self):
        return round(self.rate * 100)

    @property
    def times_average(self):
        return round(self.rate / NHANES_OVERALL_RATE, 1)


@dataclass(frozen=True)
class InsomniaCheck:
    meets_criteria: bool
    nights: int
    months_3_plus: bool
    daytime_impact: bool
    summary: str
    detail: str


def assess_apnea(snoring, sleepiness, witnessed_gasping=False):
    """Observed apnea-symptom rate for someone answering this way.

    ``witnessed_gasping`` is not part of the lookup -- it *is* the thing the
    survey counted, so someone who reports it does not need a statistic about
    how likely they are to report it.
    """
    snoring = max(0, min(3, int(snoring)))
    sleepiness = max(0, min(2, int(sleepiness)))
    rate, sample = NHANES_APNEA_RATES[(snoring, sleepiness)]

    if witnessed_gasping:
        return ApneaRisk(
            rate=rate, sample=sample, snoring=snoring, sleepiness=sleepiness,
            band="high", witnessed_gasping=True,
            headline="You have already noticed the main sign of sleep apnea",
            comparison=(
                "Someone seeing you stop breathing, gasp or choke in your sleep is "
                "the symptom clinicians screen for. That is worth raising with a "
                "doctor regardless of anything else on this page."
            ),
        )

    if rate >= 0.20:
        band = "high"
    elif rate >= 0.08:
        band = "raised"
    else:
        band = "low"

    headline = {
        "high": "Your answers match the group most likely to have sleep apnea",
        "raised": "Your answers put you above average for sleep apnea signs",
        "low": "Your answers match the group least likely to have sleep apnea",
    }[band]

    multiple = round(rate / NHANES_OVERALL_RATE, 1)
    if multiple >= 1.2:
        relative = f"about {multiple}× the {round(NHANES_OVERALL_RATE * 100)}% national average"
    elif multiple <= 0.8:
        relative = f"below the {round(NHANES_OVERALL_RATE * 100)}% national average"
    else:
        relative = f"close to the {round(NHANES_OVERALL_RATE * 100)}% national average"

    return ApneaRisk(
        rate=rate, sample=sample, snoring=snoring, sleepiness=sleepiness,
        band=band, headline=headline, witnessed_gasping=False,
        comparison=(
            f"Of the {sample:,} adults in {NHANES_CYCLE} who answered these two "
            f"questions the way you did, {round(rate * 100)}% reported snorting, "
            f"gasping or stopping breathing during sleep — {relative}."
        ),
    )


def assess_insomnia(nights_per_week, months_3_plus, daytime_impact):
    """Check reported symptoms against the standard insomnia criteria.

    Chronic insomnia disorder is defined by difficulty sleeping on at least
    three nights a week, for at least three months, with a daytime consequence.
    This is a criteria check against what the person reports -- not a
    prediction. Predicting insomnia from body measurements on the same NHANES
    data reaches ROC-AUC 0.616, which is close enough to chance not to show
    anyone.
    """
    nights = max(0, min(7, int(nights_per_week)))
    frequent = nights >= 3
    meets = frequent and bool(months_3_plus) and bool(daytime_impact)

    if meets:
        summary = "Your answers meet the usual definition of chronic insomnia"
        detail = (
            "Trouble sleeping on three or more nights a week, for three months or "
            "more, with an effect on your day is how chronic insomnia is normally "
            "defined. It is treatable, and talking therapy for insomnia (CBT-I) is "
            "usually recommended before sleeping tablets. This is worth raising "
            "with a doctor."
        )
    elif frequent and daytime_impact:
        summary = "You have frequent trouble sleeping, but not yet for three months"
        detail = (
            "Short-term insomnia often settles on its own. If it is still going at "
            "three months, or it is affecting your day badly now, see a doctor."
        )
    elif frequent:
        summary = "You have frequent trouble sleeping"
        detail = (
            "You did not report it affecting your daytime, which is part of how "
            "insomnia is defined. Worth keeping an eye on."
        )
    elif nights > 0:
        summary = "Occasional trouble sleeping"
        detail = (
            "Fewer than three nights a week is below the threshold used to define "
            "insomnia. Most people have nights like this."
        )
    else:
        summary = "No trouble sleeping reported"
        detail = "Nothing here points to insomnia."

    return InsomniaCheck(
        meets_criteria=meets, nights=nights,
        months_3_plus=bool(months_3_plus), daytime_impact=bool(daytime_impact),
        summary=summary, detail=detail,
    )
