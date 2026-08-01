"""Assert each model still beats the trivial baseline it has to beat.

A classifier on a 90.6%-negative dataset can post 90.6% accuracy by predicting
one class forever, so "87.2% accurate" was worse than useless as a headline.
Every check here is against the relevant baseline, not an absolute number.
"""

import json
from pathlib import Path

import pytest

from app.ml.bundle import MODEL_NAMES, load_metadata

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_metadata_records_provenance(name):
    meta = load_metadata(name)
    assert meta, f"{name} has no metadata.json"
    for field in ("task", "dataset", "estimator", "trained_utc", "library_versions"):
        assert field in meta, f"{name} metadata missing {field}"
    # Pickles break across scikit-learn versions; record what built them.
    assert "scikit-learn" in meta["library_versions"]


def test_heart_beats_chance_and_is_calibrated():
    m = load_metadata("heart")["metrics"]
    assert m["roc_auc"] > 0.80
    # PR-AUC against a 9.4% positive rate: chance is the prevalence itself.
    assert m["pr_auc"] > 3 * m["baseline_pr_auc"]
    assert m["balanced_accuracy"] > 0.70
    # The page shows the user a percentage, so it has to mean something:
    # mean predicted risk should track observed prevalence closely.
    assert abs(m["mean_predicted_risk"] - m["observed_prevalence"]) < 0.01
    assert m["brier_score"] < 0.09


def test_heart_threshold_is_tuned_not_hardcoded():
    meta = load_metadata("heart")
    assert "decision_threshold" in meta
    # A calibrated model on a 9.4% base rate should not be thresholded at 0.5;
    # doing so is what made balanced accuracy collapse to 0.54.
    assert 0.0 < meta["decision_threshold"] < 0.5
    assert meta["metrics"]["recall"] > 0.70


def test_sleep_is_no_longer_a_model():
    """Retraining on real national data showed no model was warranted.

    On NHANES 2017-18, snoring plus daytime sleepiness reaches ROC-AUC 0.791;
    adding age, sex, BMI, blood pressure and pulse drops it to 0.741, and an
    unfitted `2*snoring + sleepiness` matches the fitted model exactly. So the
    page reports the observed rate from the survey instead. See
    app/ml/sleep_risk.py and tests/test_sleep_risk.py.
    """
    from app.ml.features import BUILDERS

    assert "sleep" not in MODEL_NAMES
    assert "sleep" not in BUILDERS
    assert not (MODELS_DIR / "sleep").exists()


def test_migraine_beats_majority_class():
    m = load_metadata("migraine")["metrics"]
    assert m["accuracy"] > m["baseline_majority_accuracy"]
    assert m["roc_auc"] > 0.85


def test_the_lifestyle_score_is_no_longer_a_model():
    """It used to be one, fit to Gaussian noise. See tests/test_lifestyle.py.

    Guards against a model being reintroduced here without real outcome data to
    fit it to. "Health score" is a construct, not a measurable outcome.
    """
    from app.ml.features import BUILDERS

    assert "health_score" not in MODEL_NAMES
    assert "health_score" not in BUILDERS
    assert not (MODELS_DIR / "health_score").exists()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_artifacts_are_small_enough_to_commit(name):
    """A fresh clone must be able to run the app.

    Three of four models used to be gitignored for size -- the heart model alone
    was 382 MB -- so a clone could not serve predictions at all.
    """
    total = sum(f.stat().st_size for f in (MODELS_DIR / name).glob("*"))
    assert total < 5_000_000, f"{name} artifacts are {total / 1e6:.1f} MB"
