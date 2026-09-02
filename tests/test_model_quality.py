"""Assert each model still beats the trivial baseline it has to beat.

A classifier on a 90.6%-negative dataset can post 90.6% accuracy by predicting
one class forever, so "87.2% accurate" was worse than useless as a headline.
Every check here is against the relevant baseline, not an absolute number.
"""

import re
import sys
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


def test_heart_is_trained_on_the_survey_weights():
    """BRFSS rows are not people; they are people times _LLCPWT.

    Trained unweighted, the model is calibrated to whoever answers telephone
    surveys -- a group that skews old and therefore has more heart disease than
    the country does. The app quotes its percentage as literal, so it has to be
    literal about a population somebody belongs to.
    """
    meta = load_metadata("heart")
    weighting = meta.get("weighting")
    assert weighting, "heart metadata does not record any survey weighting"
    assert weighting["variable"] == "_LLCPWT"
    assert weighting["population"], "the weighting must name what it represents"

    # The correction is large enough to matter, and in the expected direction.
    # If these ever converge, the weight has stopped being applied.
    assert weighting["weighted_prevalence"] < weighting["unweighted_prevalence"]
    assert weighting["unweighted_prevalence"] - weighting["weighted_prevalence"] > 0.01


def test_heart_records_both_weighted_and_unweighted_metrics():
    """Reporting only the flattering set is how the correction gets lost."""
    meta = load_metadata("heart")
    assert "metrics_unweighted" in meta
    for key in ("roc_auc", "brier_score", "observed_prevalence"):
        assert key in meta["metrics"]
        assert key in meta["metrics_unweighted"]
    # The headline block is the weighted one, so its prevalence should track the
    # weighted figure rather than the raw sample's.
    assert abs(meta["metrics"]["observed_prevalence"]
               - meta["weighting"]["weighted_prevalence"]) < 0.01


def test_the_brfss_fetcher_carries_the_design_variables():
    """The weight has to survive the trip from the XPT to the CSV.

    _PSU is deliberately absent: every (_STSTR, _PSU) pair in the cycle is
    unique, so there is exactly one record per cluster and nothing to group by.
    """
    source = (Path(__file__).resolve().parent.parent
              / "ml_model" / "fetch_brfss.py").read_text(encoding="utf-8")
    assert "_LLCPWT" in source
    assert "SurveyWeight" in source
    assert "weighted_prevalence" in source


# --------------------------------------------------------------------------
# subgroup performance
#
# One ROC-AUC over 62,000 test rows is an average, and averages hide their
# tails. These assert that the split exists, that it is reported whether or not
# it flatters the model, and that no group has drifted into being actively
# misleading.
# --------------------------------------------------------------------------

STRATA = ("sex", "age_band", "sex_age", "race_ethnicity", "income_band", "education")

# Cells smaller than this move too much on a handful of cases to assert against.
MIN_CELL = 500


def _cells(stratum=None, min_n=MIN_CELL):
    """Every reported subgroup cell at or above ``min_n``, as (name, stats)."""
    subgroups = load_metadata("heart")["subgroups"]
    chosen = [stratum] if stratum else list(subgroups)
    return [
        (f"{s}={level}", row)
        for s in chosen
        for level, row in subgroups[s].items()
        if row["n"] >= min_n
    ]


@pytest.mark.parametrize("stratum", STRATA)
def test_heart_reports_performance_for_every_stratum(stratum):
    subgroups = load_metadata("heart")["subgroups"]
    assert stratum in subgroups, f"no subgroup audit for {stratum}"
    assert len(subgroups[stratum]) >= 2, f"{stratum} has nothing to compare"
    for level, row in subgroups[stratum].items():
        for field in ("n", "n_positive", "observed", "predicted", "brier_score"):
            assert field in row, f"{stratum}={level} is missing {field}"
        # Reported beside every figure, so a reader can see which cells are too
        # small to lean on rather than having to guess.
        assert row["n"] > 0


def test_no_subgroup_is_wildly_miscalibrated():
    """The guard. A group the model is badly wrong about must fail the build.

    Bounds are deliberately loose -- this is "something has broken", not "this
    is good". The tighter reality is in the README, which names the two groups
    the model currently serves worst.
    """
    bad = [
        (name, row["observed_over_predicted"])
        for name, row in _cells()
        if not 0.6 <= row["observed_over_predicted"] <= 1.4
    ]
    assert not bad, f"observed/predicted outside 0.6-1.4: {bad}"


def test_no_subgroup_has_collapsed_to_chance():
    """A group the model cannot rank at all is worse than no answer for it."""
    weak = [
        (name, round(row["roc_auc"], 3))
        for name, row in _cells()
        if "roc_auc" in row and row["roc_auc"] < 0.65
    ]
    assert not weak, f"ROC-AUC below 0.65: {weak}"


def test_the_oldest_band_is_the_weakest_and_stays_declared():
    """Known and named rather than averaged away.

    Discrimination falls off with age -- 0.86 at 35-49 against 0.69 past 80 --
    and past 80 is where the reader is most likely to act on the answer. If a
    retrain ever fixes this, this test fails and the README claim gets updated
    with it.
    """
    bands = load_metadata("heart")["subgroups"]["age_band"]
    assert bands["80+"]["roc_auc"] < bands["35-49"]["roc_auc"]
    assert bands["80+"]["roc_auc"] == min(
        row["roc_auc"] for row in bands.values() if "roc_auc" in row
    )


def test_reporting_strata_never_became_model_features():
    """Race, income and education are audited against, never predicted from.

    Feeding race into a clinical risk score is the mistake behind a generation
    of race-adjusted equations now being withdrawn. They are carried in the CSV
    for the subgroup split and nothing else, and this fails if one ever leaks
    into the contract.
    """
    from app.ml.features import HEART_FEATURES, HEART_RAW

    for name in ("RaceEthnicity", "IncomeBand", "EducationLevel",
                 "SurveyWeight", "Stratum"):
        assert name not in HEART_RAW, f"{name} became a raw input"
        assert name not in HEART_FEATURES, f"{name} became a feature"


def test_calibration_slope_recovers_a_known_miscalibration():
    """The slope is hand-rolled arithmetic, so it gets checked against truth.

    Simulate models whose true log-odds are half and one-and-a-half times what
    they claim, and confirm the slope reports 0.5 and 1.5.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_model"))
    from train_all import _calibration_slope

    rng = np.random.default_rng(0)
    claimed_logit = rng.normal(-2.5, 1.5, 200_000)
    claimed = 1 / (1 + np.exp(-claimed_logit))

    def outcomes(true_slope, intercept):
        true_p = 1 / (1 + np.exp(-(claimed_logit * true_slope + intercept)))
        return (rng.random(len(true_p)) < true_p).astype(float)

    assert _calibration_slope(outcomes(1.0, 0.0), claimed) == pytest.approx(1.0, abs=0.02)
    assert _calibration_slope(outcomes(0.5, -1.25), claimed) == pytest.approx(0.5, abs=0.02)
    assert _calibration_slope(outcomes(1.5, 1.25), claimed) == pytest.approx(1.5, abs=0.02)


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


# --------------------------------------------------------------------------
# BRFSS provenance
# --------------------------------------------------------------------------

def test_heart_runs_on_a_recent_documented_survey():
    """The old training file was a Kaggle derivative of BRFSS 2015 whose
    variable mapping nobody had written down, so it could never be refreshed.
    ml_model/fetch_brfss.py does the mapping in the open against CDC's release.
    """
    meta = load_metadata("heart")
    assert "BRFSS" in meta["dataset"]
    year = int(re.search(r"\b(20\d{2})\b", meta["dataset"]).group(1))
    assert year >= 2020, f"heart is still on BRFSS {year}"
    assert meta["n_rows"] > 250_000


def test_the_brfss_fetcher_verifies_its_own_mapping():
    """A miscoded variable would look like "the new cycle is just worse"
    rather than like a bug, so the fetcher checks before writing."""
    source = (Path(__file__).resolve().parent.parent
              / "ml_model" / "fetch_brfss.py").read_text(encoding="utf-8")
    assert "def verify(" in source
    for check in ("prevalence", "median BMI", "is not binary"):
        assert check in source


def test_fruit_and_veg_are_gone_from_the_contract():
    """They moved ROC-AUC by 0.0001 and were the only thing pinning the model
    to the 2015 cycle, which is the last one that asked."""
    from app.ml.features import HEART_RAW

    assert "Fruits" not in HEART_RAW
    assert "Veggies" not in HEART_RAW


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_migraine_declares_that_its_source_is_undocumented():
    """The one input in this repository that cannot name where it came from.

    Recorded in the artifact rather than only in the README, so the page can
    read it and say so. A caveat the reader never sees is a caveat the project
    has made to itself.
    """
    provenance = load_metadata("migraine").get("provenance")
    assert provenance, "migraine metadata does not record its provenance"
    assert provenance["documented"] is False
    assert provenance["note"] and provenance["consequence"]


def test_the_documented_models_do_not_carry_the_warning():
    """It must mean something. If every model declared undocumented provenance
    the notice would be wallpaper."""
    heart = load_metadata("heart")
    assert not heart.get("provenance", {}).get("documented") is False
    assert "BRFSS" in heart["dataset"]


def test_migraine_records_whether_its_percentage_means_anything():
    """The page prints a confidence to one decimal place. Heart has carried
    calibration figures since it was rebuilt; migraine printed a number with
    nothing anywhere saying whether it was literal.

    It is, as it turns out -- mean predicted risk tracks observed prevalence
    closely. That was not knowable before it was measured.
    """
    m = load_metadata("migraine")["metrics"]
    for key in ("brier_score", "mean_predicted_risk", "observed_prevalence",
                "calibration_slope"):
        assert key in m, f"migraine metrics missing {key}"
    assert abs(m["mean_predicted_risk"] - m["observed_prevalence"]) < 0.05


# --------------------------------------------------------------------------
# how precise the number on the page actually is
# --------------------------------------------------------------------------

def test_heart_records_what_happened_in_each_risk_band():
    """The page prints a percentage, and a percentage with no width reads as a
    measurement. It was being printed to two decimal places from a model whose
    Brier score is 0.057."""
    bins = load_metadata("heart").get("risk_bins")
    assert bins, "heart metadata records no risk bands"
    for band in bins:
        for field in ("low", "high", "n", "effective_n", "mean_predicted",
                      "observed", "observed_low", "observed_high"):
            assert field in band, f"band {band.get('low')} is missing {field}"
        assert band["observed_low"] <= band["observed"] <= band["observed_high"], (
            f"band {band['low']}-{band['high']} has its own rate outside its "
            f"interval: {band['observed']:.3f} vs "
            f"({band['observed_low']:.3f}, {band['observed_high']:.3f})"
        )


def test_the_bands_cover_every_possible_prediction():
    """A reader whose risk falls in no band silently gets no interval."""
    bins = sorted(load_metadata("heart")["risk_bins"], key=lambda b: b["low"])
    assert bins[0]["low"] == 0.0
    assert bins[-1]["high"] >= 1.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        assert lower["high"] == upper["low"], "the bands leave a gap"


def test_the_intervals_use_the_effective_sample_size():
    """Weighted rates need weighted intervals. Computing the width from raw
    counts put the point estimate outside its own interval on the first run --
    the 2-5% band read "3.6% (2.9-3.5)".

    Unequal weights always make Kish's effective n smaller than the raw count,
    so this also asserts the correction is actually being applied.
    """
    for band in load_metadata("heart")["risk_bins"]:
        assert band["effective_n"] < band["n"], (
            f"band {band['low']}-{band['high']} has effective n "
            f"{band['effective_n']} >= raw n {band['n']}"
        )
