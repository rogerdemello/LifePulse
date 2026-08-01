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


def test_sleep_predicts_three_classes_including_healthy():
    """The previous model had two classes and told every user they were ill."""
    meta = load_metadata("sleep")
    assert meta["classes"] == ["None", "Insomnia", "Sleep Apnea"]
    m = meta["metrics"]
    assert m["accuracy"] > m["baseline_majority_accuracy"]
    assert m["balanced_accuracy"] > 0.70
    # Every class must be genuinely predicted, not just the majority.
    for label in meta["classes"]:
        assert m["per_class"][label]["recall"] > 0.4, f"{label} recall too low"


def test_migraine_beats_majority_class():
    m = load_metadata("migraine")["metrics"]
    assert m["accuracy"] > m["baseline_majority_accuracy"]
    assert m["roc_auc"] > 0.85


def test_health_score_beats_a_plain_linear_fit():
    """The engineered features have to earn their keep.

    The model this replaced was a 4.8 MB RandomForest that scored *below* a
    plain linear regression on the seven raw columns.
    """
    m = load_metadata("health_score")["metrics"]
    assert m["r2"] >= m["baseline_linear_on_raw_features_r2"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_artifacts_are_small_enough_to_commit(name):
    """A fresh clone must be able to run the app.

    Three of four models used to be gitignored for size -- the heart model alone
    was 382 MB -- so a clone could not serve predictions at all.
    """
    total = sum(f.stat().st_size for f in (MODELS_DIR / name).glob("*"))
    assert total < 5_000_000, f"{name} artifacts are {total / 1e6:.1f} MB"
