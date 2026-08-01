"""Turning a result into questions worth asking a doctor.

A percentage is not consultable. "Is 12% high for someone my age?" is. This
module builds the short list of questions that goes on the printable visit
summary, driven by what the assessment actually found rather than a generic
leaflet.

Two rules shape everything here:

* Questions, never instructions. This app is not qualified to tell anyone to
  start a medication or skip a test. It is qualified to help someone walk into
  an appointment knowing what to ask.
* Grounded in the specific result. A question that would have been generated
  regardless of the answers is filler, and filler is what makes people stop
  reading the parts that matter.
"""

from __future__ import annotations

# Follow-up prompts keyed on the raw field that drove the result. Only fields
# a person can actually act on or ask about appear here -- there is no useful
# question to ask about your own sex or age.
FACTOR_QUESTIONS = {
    "BMI": "My BMI is {value}. Is that a concern for me specifically, and what would a realistic target look like?",
    "HighBP": "How often should I be checking my blood pressure at home, and what reading should prompt me to call?",
    "HighChol": "When was my cholesterol last measured, and is it due again?",
    "Smoker": "What support for stopping smoking is available to me through this practice?",
    "Diabetes": "Should my blood sugar be monitored more closely than it is now?",
    "PhysActivity": "What amount and kind of activity would be safe for me to start with?",
    "GenHlth": "I rated my general health as {value}. What might be behind that, and is it worth investigating?",
    "MentHlth": "I've had {value} of poor mental health this month. Can we talk about that too?",
    "PhysHlth": "I've had {value} of poor physical health this month. Is that worth looking into?",
    "DiffWalk": "I have difficulty walking. Could that be treatable, or is it worth a referral?",
    "Stroke": "Given my history of stroke, is my current prevention plan still the right one?",
    "HvyAlcoholConsump": "Is my alcohol intake affecting anything you can see in my results?",
    "Systolic": "My blood pressure reading was {value}. Should it be re-checked here?",
    "Diastolic": "My blood pressure reading was {value}. Should it be re-checked here?",
    "Heart Rate": "My resting heart rate is {value}. Is that normal for someone like me?",
    "Sleep Duration": "I'm sleeping {value} a night. Could that be causing symptoms I've noticed?",
    "Sleep Hours": "I'm sleeping {value} a night. Could that be causing symptoms I've noticed?",
    "Sleep_Hours": "I'm sleeping {value} a night. Could that be causing symptoms I've noticed?",
    "Quality of Sleep": "My sleep quality is poor even when I get enough hours. Is that worth investigating?",
    "Stress Level": "Stress came out as a major factor for me. What support is available?",
    "Stress": "Stress came out as a major factor for me. What support is available?",
    "Daily Steps": "I average {value} steps a day. Is increasing that safe for me?",
    "Caffeine": "I have {value} caffeinated drinks a day. Could cutting down help?",
    "Water Intake": "Could dehydration be contributing to my symptoms?",
    "Screen Time": "Could my screen time be contributing to this?",
    "Skipped Meals": "I skip meals fairly often. Could that be a trigger?",
    "Diet_Quality": "What one change to my diet would make the most difference?",
    "Exercise_Frequency": "What amount and kind of activity would be safe for me to start with?",
    "Smoking_Status": "What support for stopping smoking is available to me through this practice?",
    "Alcohol_Consumption": "Is my alcohol intake affecting anything you can see in my results?",
}

# The question that leads each summary, phrased around what the tool reported.
OPENERS = {
    "heart": "An online screening tool put my heart-disease risk at {headline}. "
             "Does that match what you'd expect for someone with my history?",
    "sleep": "An online screening tool suggested {headline}. Is that worth "
             "investigating properly, and would a sleep study be appropriate?",
    "migraine": "An online screening tool flagged {headline}. Could we look at "
                "what's triggering my headaches?",
    "health_score": "I've been tracking my lifestyle factors and wanted to check "
                    "which of them matter most for someone with my history.",
}


TOPICS = {"heart": "heart disease risk", "migraine": "migraine risk"}


def questions_for(model_name, headline, factors, caveats=(), flags=(), band=None):
    """Build the "questions to ask" list for one assessment.

    Red flags come first and are phrased as things to raise immediately;
    factor-driven questions follow, in the order the factors mattered.

    When ``band`` is given and app/ml/phrasings.json holds pre-written
    questions for it, those are used ahead of the templates below. That file is
    generated offline (ml_model/generate_phrasings.py), so better wording costs
    no per-request call and sends nothing about the user anywhere.
    """
    from app.ml.phrasings import questions_for_band

    questions = []

    for flag in flags:
        if flag.urgency == "emergency":
            questions.append(
                f"I need to ask about this first: {flag.title.lower()}."
            )
        else:
            questions.append(f"Should we talk about {flag.title.lower()}?")

    opener = OPENERS.get(model_name)
    if opener:
        questions.append(opener.format(headline=headline))

    # Before the factor questions, not after. When the model admits it has no
    # evidence for someone like you, that undercuts everything below it -- and
    # trailing it after four factor questions meant the six-question cap
    # silently dropped it.
    if caveats:
        questions.append(
            "This tool said some of my answers were outside the range it was "
            "built on, so its estimate may not apply to me. Is there a proper "
            "assessment we should do instead?"
        )

    if band:
        questions.extend(questions_for_band(TOPICS.get(model_name, model_name), band))

    for factor in factors:
        template = FACTOR_QUESTIONS.get(factor.field)
        if not template:
            continue
        question = template.format(value=factor.value)
        if question not in questions:
            questions.append(question)

    return questions[:6]
