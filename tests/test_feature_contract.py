"""The regression guard for the bug this whole layer exists to prevent.

Every route used to build its model's features by hand, and the loader filled
in 0.0 for anything it could not find. Names drifted, nothing raised, and all
four models were served feature vectors far outside their training
distributions -- sleep was feeding `Blood_Pressure_Mean` at z = -16.7.

These tests assert the two halves still agree.
"""

import json
from pathlib import Path

import pytest

from app.ml import features as F
from app.ml.bundle import MODEL_NAMES, ModelBundle, get_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_features_match_the_code(name):
    """The trained artifact's contract must equal what app.ml.features defines.

    If this fails, the models were trained against a different version of the
    feature code: retrain with `python ml_model/train_all.py`.
    """
    saved = json.loads((MODELS_DIR / name / "features.json").read_text("utf-8"))
    assert saved == F.FEATURES[name]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_builder_output_matches_contract_exactly(name):
    """Builder columns must equal the contract in the same order.

    Order matters: the scaler and the estimator index by position, so a correct
    set of columns in the wrong order silently mispredicts.
    """
    built = F.BUILDERS[name](SAMPLE_RAW[name])
    assert list(built.columns) == F.FEATURES[name]
    assert len(built) == 1
    assert built.notna().all().all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_missing_raw_field_raises_instead_of_defaulting(name):
    """A dropped field must raise, not quietly become 0.0.

    This is the exact failure that made every prediction wrong: the old loader
    used `input_data.get(feature, 0)`.
    """
    incomplete = dict(SAMPLE_RAW[name])
    dropped = F.RAW_FIELDS[name][0]
    del incomplete[dropped]

    with pytest.raises(F.FeatureContractError) as excinfo:
        F.BUILDERS[name](incomplete)
    assert dropped in str(excinfo.value)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_unrecognised_category_raises(name):
    """Garbage in a categorical field must raise rather than be coerced."""
    payload = dict(SAMPLE_RAW[name])
    categorical = next(
        (f for f in F.RAW_FIELDS[name] if isinstance(payload[f], str)), None
    )
    if categorical is None:
        pytest.skip(f"{name} has no categorical raw fields")
    payload[categorical] = "definitely-not-a-valid-category"
    with pytest.raises(F.FeatureContractError):
        F.BUILDERS[name](payload)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_bundle_loads_and_predicts(name):
    bundle = get_model(name)
    assert isinstance(bundle, ModelBundle)
    assert bundle.features == F.FEATURES[name]
    assert len(bundle.predict(SAMPLE_RAW[name])) == 1


def test_bundle_rejects_a_stale_contract(monkeypatch):
    """Loading must fail loudly if the code and the artifact disagree."""
    from app.ml import bundle as bundle_module

    monkeypatch.setitem(bundle_module.FEATURES, "heart", ["not", "the", "real", "features"])
    with pytest.raises(bundle_module.ModelNotAvailable, match="different feature set"):
        ModelBundle.load("heart")


# --------------------------------------------------------------------------
# One realistic record per model, in raw (pre-engineering) form.
# --------------------------------------------------------------------------

SAMPLE_RAW = {
    "heart": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31.0, "Smoker": 1,
        "Stroke": 0, "Diabetes": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
        "HvyAlcoholConsump": 0, "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
        "DiffWalk": 1, "Sex": 1, "Age": 10,
    },
    "sleep": {
        "Gender": "Female", "Age": 42, "Sleep Duration": 6.2,
        "Quality of Sleep": 6, "Physical Activity Level": 45,
        "Stress Level": 6, "Heart Rate": 76, "Daily Steps": 6000,
        "BMI Category": "Overweight", "Systolic": 132, "Diastolic": 87,
    },
    "migraine": {
        "Age": 34, "Gender": "Female", "Sleep Hours": 5.5, "Water Intake": 3,
        "Skipped Meals": "Yes", "Caffeine": 4, "Stress": 8, "Screen Time": 9,
        "Physical Activity": "1-2 days/week", "Menstruating": "Yes",
    },
    "health_score": {
        "Age": 38, "BMI": 26.5, "Exercise_Frequency": 3, "Diet_Quality": 70.0,
        "Sleep_Hours": 7.0, "Smoking_Status": 0, "Alcohol_Consumption": 2.0,
    },
}
