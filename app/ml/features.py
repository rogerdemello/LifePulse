"""Single source of truth for feature engineering.

Both sides of the pipeline import from here:

    ml_model/train_all.py   ->  fits scalers/models on build_*(csv)
    app/ml/bundle.py        ->  serves predictions on build_*(form_dict)

That is the whole point of this module. Previously each route reimplemented
its model's feature engineering by hand, the names drifted, and the loader
silently substituted 0.0 for anything it could not find -- which the scaler
then turned into an extreme z-score. Every prediction the app made was
computed far outside the training distribution.

Rules for anyone editing this file:

1. A model's ``*_FEATURES`` list is its contract. The builder must return
   exactly those columns, in exactly that order. ``tests/test_feature_contract.py``
   enforces it.
2. Builders take *raw* fields -- the same names the CSV uses -- and accept both
   a single dict (one request) and a DataFrame (a training set). Never let the
   serving path and the training path diverge into separate code.
3. Never feed the model a value the user did not supply. If a feature cannot be
   derived from the form, drop it from the contract rather than defaulting it.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


class FeatureContractError(ValueError):
    """Raised when raw input cannot produce the exact contracted feature set."""

    def __init__(self, model, missing=(), unexpected=(), detail=""):
        self.model = model
        self.missing = list(missing)
        self.unexpected = list(unexpected)
        parts = [f"{model}: feature contract violated"]
        if self.missing:
            parts.append(f"missing {self.missing}")
        if self.unexpected:
            parts.append(f"unexpected {self.unexpected}")
        if detail:
            parts.append(detail)
        super().__init__("; ".join(parts))


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _as_frame(raw):
    """Accept a dict, Series, or DataFrame and return a DataFrame."""
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, pd.Series):
        return raw.to_frame().T.copy()
    if isinstance(raw, dict):
        return pd.DataFrame([raw])
    raise TypeError(f"expected dict, Series, or DataFrame, got {type(raw).__name__}")


def _require(df, cols, model):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FeatureContractError(model, missing=missing)


def _finish(df, features, model):
    """Select the contracted columns, in order, and verify nothing is missing.

    This is the check that the old ``input_data.get(f, 0)`` skipped.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise FeatureContractError(model, missing=missing)
    out = df[list(features)].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise FeatureContractError(
            model, detail=f"non-numeric or missing values in {bad}"
        )
    return out.astype("float64")


def _norm_text(value):
    """Normalise a category label for lookup.

    Collapses unicode dashes to ASCII '-', strips accents/whitespace, lowercases.
    The migraine dataset ships '3–5 days/week' with an en-dash while the old
    training script mapped the ASCII '3-5 days/week'; 1,686 of 2,000 rows became
    NaN and were filled with a constant, which killed the feature outright.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def _categorical(series, mapping, model, column, na_value=None):
    """Map a categorical column, tolerating strings or already-encoded numbers.

    ``mapping`` keys are matched after ``_norm_text`` normalisation. Values that
    are already valid codes pass through. Anything unrecognised raises rather
    than silently becoming NaN.
    """
    lookup = {_norm_text(k): v for k, v in mapping.items()}
    valid = set(mapping.values())

    def convert(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if na_value is None:
                raise FeatureContractError(
                    model, detail=f"{column}: missing value with no default"
                )
            return na_value
        key = _norm_text(value)
        if key in lookup:
            return lookup[key]
        try:
            code = int(float(value))
        except (TypeError, ValueError):
            code = None
        if code is not None and code in valid:
            return code
        raise FeatureContractError(
            model,
            detail=f"{column}: unrecognised value {value!r} "
                   f"(expected one of {sorted(mapping)})",
        )

    return series.map(convert).astype("float64")


def _bmi_category(bmi):
    """WHO bands as ordinal codes: 0 under, 1 normal, 2 over, 3 obese.

    Verified against the shipped heart scaler: these bands reproduce the stored
    one-hot means (0.0123/0.2718/0.3696/0.3463 vs 0.0125/0.2717/0.3693/0.3465).
    """
    return pd.cut(
        pd.to_numeric(bmi, errors="coerce"),
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype("float64")


# --------------------------------------------------------------------------
# heart disease  --  data/heart_disease_health_indicators_BRFSS2015.csv
# --------------------------------------------------------------------------

HEART_RAW = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "HvyAlcoholConsump", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
]

# Education and Income are absent because the form never collected them: the
# old route hardcoded 4 and 5 for every user, so the model saw a constant.
#
# Fruits and Veggies are absent because they earned nothing. Dropping both moves
# ROC-AUC from 0.8485 to 0.8486, and BRFSS stopped running that module after
# 2015 -- so two questions nobody needed to answer were also the only thing
# tying the model to a decade-old survey.
HEART_FEATURES = HEART_RAW + [
    "Health_Score", "Lifestyle_Score", "Risk_Count", "Age_BMI",
    "BMI_Category_0", "BMI_Category_1", "BMI_Category_2", "BMI_Category_3",
]


def brfss_age_bucket(years):
    """Convert age in years to the BRFSS ``_AGEG5YR`` bucket (1-13).

    1 = 18-24, 2 = 25-29, 3 = 30-34, ... 12 = 75-79, 13 = 80+.

    The heart form asks for age in years while the model was trained on these
    buckets, and the route used to pass the years straight through. Because
    every real age (25-120) sits beyond the model's highest split at 13, all of
    them landed in the same leaf: a 25-year-old and an 80-year-old both scored
    15.73%. Age -- the strongest single predictor of cardiovascular risk -- was
    doing nothing at all. Converted properly the same profile spans 0.22% to
    16.00%.
    """
    years = float(years)
    if years < 25:
        return 1
    return min(13, 2 + int((years - 25) // 5))


def build_heart(raw):
    """BRFSS heart-disease features. ``Age`` is the BRFSS 1-13 bucket, not years."""
    df = _as_frame(raw)
    _require(df, HEART_RAW, "heart")
    for col in HEART_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Health_Score"] = df["GenHlth"] + df["PhysHlth"] + df["MentHlth"]
    df["Lifestyle_Score"] = (
        df["PhysActivity"] - df["Smoker"] - df["HvyAlcoholConsump"]
    )
    df["Risk_Count"] = (
        df["HighBP"] + df["HighChol"] + df["Smoker"] + df["Stroke"]
        + (df["Diabetes"] > 0).astype("float64")
    )
    df["Age_BMI"] = df["Age"] * df["BMI"]

    band = _bmi_category(df["BMI"])
    for code in (0, 1, 2, 3):
        df[f"BMI_Category_{code}"] = (band == code).astype("float64")

    return _finish(df, HEART_FEATURES, "heart")


# --------------------------------------------------------------------------
# sleep disorder  --  data/Sleep_health_and_lifestyle_dataset (1).csv
# --------------------------------------------------------------------------

SLEEP_RAW = [
    "Gender", "Age", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps",
    "BMI Category",
]

# Occupation is deliberately absent: the healthy rate is flat across all 11
# values (0.656-0.727 against a 0.70 base rate), so it carries no signal, and
# the old route defaulted every user to 'Nurse' anyway.
SLEEP_FEATURES = [
    "Gender", "Age", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps",
    "BMI_Category", "Systolic", "Diastolic", "Blood_Pressure_Mean",
    "Pulse_Pressure", "Sleep_Quality_Ratio", "Activity_Stress_Ratio",
    "Steps_per_Hour_Awake", "Heart_Stress_Product", "Sleep_Deficit",
]

SLEEP_CLASSES = ["None", "Insomnia", "Sleep Apnea"]

_SLEEP_GENDER = {"female": 0, "male": 1}
# 'Normal Weight' and 'Normal' are the same band recorded two ways in the source.
_SLEEP_BMI = {
    "normal": 0, "normal weight": 0, "overweight": 1, "obese": 2,
    "underweight": 0,
}


def parse_blood_pressure(df, model="sleep"):
    """Return (systolic, diastolic) from either a '120/80' string or two columns."""
    if "Systolic" in df.columns and "Diastolic" in df.columns:
        sys_ = pd.to_numeric(df["Systolic"], errors="coerce")
        dia = pd.to_numeric(df["Diastolic"], errors="coerce")
    elif "Blood Pressure" in df.columns:
        parts = df["Blood Pressure"].astype(str).str.strip().str.split("/", expand=True)
        if parts.shape[1] < 2:
            raise FeatureContractError(
                model, detail="Blood Pressure must look like '120/80'"
            )
        sys_ = pd.to_numeric(parts[0], errors="coerce")
        dia = pd.to_numeric(parts[1], errors="coerce")
    else:
        raise FeatureContractError(
            model, missing=["Blood Pressure (or Systolic + Diastolic)"]
        )
    if sys_.isna().any() or dia.isna().any():
        raise FeatureContractError(model, detail="blood pressure is not numeric")
    return sys_, dia


def build_sleep(raw):
    """Sleep-disorder features.

    The raw blood-pressure string is parsed into numbers instead of being
    label-encoded. The old encoder knew exactly 8 literal readings
    ('115/75'...'132/87') and rejected everything else.
    """
    df = _as_frame(raw)
    _require(df, SLEEP_RAW, "sleep")

    df["Gender"] = _categorical(df["Gender"], _SLEEP_GENDER, "sleep", "Gender")
    df["BMI_Category"] = _categorical(
        df["BMI Category"], _SLEEP_BMI, "sleep", "BMI Category"
    )
    for col in ["Age", "Sleep Duration", "Quality of Sleep",
                "Physical Activity Level", "Stress Level", "Heart Rate",
                "Daily Steps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Systolic"], df["Diastolic"] = parse_blood_pressure(df)
    df["Blood_Pressure_Mean"] = (df["Systolic"] + df["Diastolic"]) / 2.0
    df["Pulse_Pressure"] = df["Systolic"] - df["Diastolic"]

    df["Sleep_Quality_Ratio"] = df["Quality of Sleep"] / df["Sleep Duration"]
    df["Activity_Stress_Ratio"] = (
        df["Physical Activity Level"] / (df["Stress Level"] + 1)
    )
    df["Steps_per_Hour_Awake"] = df["Daily Steps"] / (24 - df["Sleep Duration"])
    df["Heart_Stress_Product"] = df["Heart Rate"] * df["Stress Level"]
    df["Sleep_Deficit"] = 8.0 - df["Sleep Duration"]

    return _finish(df, SLEEP_FEATURES, "sleep")


# --------------------------------------------------------------------------
# migraine  --  data/migraine_dataset_500 (1).csv
# --------------------------------------------------------------------------

MIGRAINE_RAW = [
    "Age", "Gender", "Sleep Hours", "Water Intake", "Skipped Meals",
    "Caffeine", "Stress", "Screen Time", "Physical Activity", "Menstruating",
]

MIGRAINE_FEATURES = MIGRAINE_RAW + [
    "Sleep_Stress", "Water_Caffeine", "Activity_Stress_Ratio",
    "Screen_Sleep_Ratio", "Dehydration_Risk", "Sleep_Quality",
    "High_Risk_Combo", "Stress_Squared", "Water_Squared", "Sleep_Squared",
]

_MIGRAINE_GENDER = {"female": 0, "male": 1}
_MIGRAINE_YESNO = {"no": 0, "yes": 1}
_MIGRAINE_MENSTRUATING = {"no": 0, "yes": 1, "not applicable": 2}
# Source labels use an en-dash; _norm_text folds it to ASCII before lookup.
_MIGRAINE_ACTIVITY = {
    "none": 0, "1-2 days/week": 1, "3-5 days/week": 2, "daily": 3,
}


def build_migraine(raw):
    """Migraine-risk features.

    ``Physical Activity`` is mapped through ``_norm_text`` so the dataset's
    en-dash labels resolve. A blank value means no activity (code 0) -- the 417
    blank rows in the source are absences, not unknowns.
    """
    df = _as_frame(raw)
    _require(df, MIGRAINE_RAW, "migraine")

    df["Gender"] = _categorical(df["Gender"], _MIGRAINE_GENDER, "migraine", "Gender")
    df["Skipped Meals"] = _categorical(
        df["Skipped Meals"], _MIGRAINE_YESNO, "migraine", "Skipped Meals"
    )
    df["Menstruating"] = _categorical(
        df["Menstruating"], _MIGRAINE_MENSTRUATING, "migraine", "Menstruating"
    )
    df["Physical Activity"] = _categorical(
        df["Physical Activity"], _MIGRAINE_ACTIVITY, "migraine",
        "Physical Activity", na_value=0,
    )
    for col in ["Age", "Sleep Hours", "Water Intake", "Caffeine", "Stress",
                "Screen Time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sleep_Stress"] = df["Sleep Hours"] * df["Stress"]
    df["Water_Caffeine"] = df["Water Intake"] / (df["Caffeine"] + 1)
    df["Activity_Stress_Ratio"] = df["Physical Activity"] / (df["Stress"] + 1)
    df["Screen_Sleep_Ratio"] = df["Screen Time"] / (df["Sleep Hours"] + 1)
    df["Dehydration_Risk"] = (
        (df["Caffeine"] > 3) & (df["Water Intake"] < 4)
    ).astype("float64")
    df["Sleep_Quality"] = (
        (df["Sleep Hours"] >= 6) & (df["Sleep Hours"] <= 9)
    ).astype("float64")
    df["High_Risk_Combo"] = (
        (df["Stress"] > 7) & (df["Water Intake"] < 4)
    ).astype("float64")
    df["Stress_Squared"] = df["Stress"] ** 2
    df["Water_Squared"] = df["Water Intake"] ** 2
    df["Sleep_Squared"] = df["Sleep Hours"] ** 2

    return _finish(df, MIGRAINE_FEATURES, "migraine")


# --------------------------------------------------------------------------
# health score  --  data/synthetic_health_data.csv
# --------------------------------------------------------------------------

HEALTH_RAW = [
    "Age", "BMI", "Exercise_Frequency", "Diet_Quality", "Sleep_Hours",
    "Smoking_Status", "Alcohol_Consumption",
]

HEALTH_FEATURES = HEALTH_RAW + [
    "Smoke_Alcohol", "Exercise_per_Age", "Exercise_Diet", "Sleep_Alcohol",
    "BMI_squared", "Age_squared", "Sleep_squared", "Exercise_squared",
    "Sleep_Deviation", "BMI_Deviation",
]


def build_health(raw):
    """Health-score features.

    ``Sleep_Deviation`` and ``BMI_Deviation`` carry the U-shaped penalties
    (too little *and* too much sleep is bad) that a linear model cannot
    otherwise express.
    """
    df = _as_frame(raw)
    _require(df, HEALTH_RAW, "health_score")
    for col in HEALTH_RAW:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Smoke_Alcohol"] = df["Smoking_Status"] * df["Alcohol_Consumption"]
    df["Exercise_per_Age"] = df["Exercise_Frequency"] / (df["Age"] + 1)
    df["Exercise_Diet"] = df["Exercise_Frequency"] * df["Diet_Quality"]
    df["Sleep_Alcohol"] = df["Sleep_Hours"] * df["Alcohol_Consumption"]
    df["BMI_squared"] = df["BMI"] ** 2
    df["Age_squared"] = df["Age"] ** 2
    df["Sleep_squared"] = df["Sleep_Hours"] ** 2
    df["Exercise_squared"] = df["Exercise_Frequency"] ** 2
    df["Sleep_Deviation"] = (df["Sleep_Hours"] - 8.0).abs()
    df["BMI_Deviation"] = (df["BMI"] - 22.0).abs()

    return _finish(df, HEALTH_FEATURES, "health_score")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

# Only heart and migraine are models. The lifestyle score is a rubric
# (app/ml/lifestyle.py) and sleep is an empirical lookup over real survey data
# (app/ml/sleep_risk.py) -- in both cases the evidence said a model earned
# nothing. build_health and build_sleep remain for anyone refitting them
# against better data; nothing in the app calls them.
BUILDERS = {
    "heart": build_heart,
    "migraine": build_migraine,
}

FEATURES = {
    "heart": HEART_FEATURES,
    "migraine": MIGRAINE_FEATURES,
}

RAW_FIELDS = {
    "heart": HEART_RAW,
    "migraine": MIGRAINE_RAW,
}


# --------------------------------------------------------------------------
# human-readable labels
#
# Raw field names are dataset conventions ("GenHlth", "HvyAlcoholConsump").
# Result pages explain which answers drove an outcome, and "HvyAlcoholConsump
# raised your risk" is not an explanation anyone can act on.
# --------------------------------------------------------------------------

FIELD_LABELS = {
    # heart (BRFSS)
    "HighBP": "high blood pressure",
    "HighChol": "high cholesterol",
    "CholCheck": "cholesterol checked recently",
    "BMI": "BMI",
    "Smoker": "smoking history",
    "Stroke": "history of stroke",
    "Diabetes": "diabetes",
    "PhysActivity": "physical activity",
    "Fruits": "eating fruit daily",
    "Veggies": "eating vegetables daily",
    "HvyAlcoholConsump": "heavy alcohol use",
    "GenHlth": "self-rated general health",
    "MentHlth": "days of poor mental health",
    "PhysHlth": "days of poor physical health",
    "DiffWalk": "difficulty walking",
    "Sex": "sex",
    "Age": "age",
    # sleep
    "Gender": "gender",
    "Sleep Duration": "sleep duration",
    "Quality of Sleep": "sleep quality",
    "Physical Activity Level": "physical activity level",
    "Stress Level": "stress level",
    "Heart Rate": "resting heart rate",
    "Daily Steps": "daily steps",
    "BMI Category": "BMI category",
    "Systolic": "systolic blood pressure",
    "Diastolic": "diastolic blood pressure",
    # migraine
    "Sleep Hours": "sleep hours",
    "Water Intake": "water intake",
    "Skipped Meals": "skipping meals",
    "Caffeine": "caffeine intake",
    "Stress": "stress",
    "Screen Time": "screen time",
    "Physical Activity": "physical activity",
    "Menstruating": "menstruation",
    # health score
    "Exercise_Frequency": "exercise frequency",
    "Diet_Quality": "diet quality",
    "Sleep_Hours": "sleep hours",
    "Smoking_Status": "smoking",
    "Alcohol_Consumption": "alcohol consumption",
}


def label_for(field_name):
    """Human-readable name for a raw input field."""
    return FIELD_LABELS.get(field_name, field_name.replace("_", " ").lower())


# Coded answers rendered back in the words the form used. "high blood
# pressure: 1" explains nothing; "high blood pressure: yes" explains it.
_YES_NO = {0: "no", 1: "yes"}

VALUE_LABELS = {
    "HighBP": _YES_NO, "HighChol": _YES_NO, "CholCheck": _YES_NO,
    "Smoker": _YES_NO, "Stroke": _YES_NO, "PhysActivity": _YES_NO,
    "Fruits": _YES_NO, "Veggies": _YES_NO, "HvyAlcoholConsump": _YES_NO,
    "DiffWalk": _YES_NO, "Smoking_Status": _YES_NO,
    "Diabetes": {0: "no", 1: "prediabetes", 2: "yes"},
    "Sex": {0: "female", 1: "male"},
    "GenHlth": {1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"},
    "Menstruating": {0: "no", 1: "yes", 2: "not applicable"},
    "Physical Activity": {
        0: "none", 1: "1-2 days/week", 2: "3-5 days/week", 3: "daily",
    },
    "Skipped Meals": _YES_NO,
    # Heart's Age reaches the model as a BRFSS 5-year bucket; show the years back.
    "Age": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
        12: "75-79", 13: "80+",
    },
}

# Age is a bucket for heart and plain years everywhere else, so the lookup
# above must only apply to heart.
_MODEL_SCOPED_VALUE_LABELS = {"Age": {"heart"}}

_UNIT_SUFFIXES = {
    "MentHlth": " days", "PhysHlth": " days",
    "Systolic": " mmHg", "Diastolic": " mmHg",
    "Heart Rate": " bpm",
    "Sleep Duration": " hrs", "Sleep Hours": " hrs", "Sleep_Hours": " hrs",
    "Screen Time": " hrs",
}


def describe_value(field_name, value, model_name=None):
    """Render an answer the way the person gave it, not the way the model sees it."""
    scope = _MODEL_SCOPED_VALUE_LABELS.get(field_name)
    if field_name in VALUE_LABELS and (scope is None or model_name in scope):
        try:
            return VALUE_LABELS[field_name][int(float(value))]
        except (TypeError, ValueError, KeyError):
            pass
    try:
        return f"{float(value):g}{_UNIT_SUFFIXES.get(field_name, '')}"
    except (TypeError, ValueError):
        return str(value)
