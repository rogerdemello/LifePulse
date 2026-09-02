"""Train every LifePulse model from the CSVs in ``data/``.

Run from the repository root::

    python ml_model/train_all.py                 # all three
    python ml_model/train_all.py --model heart   # just one

Feature engineering lives in ``app/ml/features.py`` and is imported, never
duplicated here. That is deliberate: the previous training scripts each carried
their own copy of the feature code, the serving routes carried a third, and the
three drifted apart until every prediction the app made was wrong.

Each model writes four files to ``app/models/<name>/``: ``model.joblib``,
``scaler.joblib``, ``features.json``, ``metadata.json``.

Every model is scored against the trivial baseline it has to beat. A classifier
on a 90% negative dataset can post 90% accuracy by predicting one class, so
accuracy alone is reported only alongside that baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml import features as F

DATA = ROOT / "data"
MODELS = ROOT / "app" / "models"
SEED = 42

log = logging.getLogger("train")


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

ARTIFACTS = ("model.joblib", "scaler.joblib", "features.json", "metadata.json")


def _save(name, model, scaler, feature_names, metadata):
    out = MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    joblib.dump(scaler, out / "scaler.joblib", compress=3)
    (out / "features.json").write_text(json.dumps(list(feature_names), indent=2), "utf-8")

    metadata = dict(metadata)
    metadata.update(
        trained_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_features=len(feature_names),
        library_versions={
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")

    size_mb = sum((out / f).stat().st_size for f in ARTIFACTS) / 1e6
    log.info("  saved -> app/models/%s/  (%.2f MB)", name, size_mb)
    return metadata


def _weighted_median(values, weights):
    """The value at which half the *population's* weight lies below."""
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    crossing = np.searchsorted(np.cumsum(weights), weights.sum() / 2.0)
    return float(values[min(crossing, len(values) - 1)])


def _profile(df, fields, weights=None):
    """Summarise the raw inputs the model actually saw during training.

    Two things downstream depend on this, and -- this is the part worth getting
    right -- they want the summary computed two different ways:

    * **Extrapolation warnings** use ``min``/``p1``/``p99``/``max``. The question
      there is "did the model ever see a value like this?", so these stay
      unweighted however the sample was drawn. The sleep dataset contained no
      systolic reading above 144; that was true of the rows, and survey weights
      would not have made it less true.
    * **Result explanations** use ``median``/``mode``. They re-predict with one
      field swapped for its typical value, and the sentence the user reads says
      "compared with a typical person". That is a claim about the population, so
      when a survey weight is available it is used. On BRFSS it moves the
      typical age band down by a decade, because the unweighted survey
      over-represents older respondents.

    ``weights`` is optional and everything falls back to the unweighted summary
    without it, which is what the migraine dataset gets -- it has no design.
    """
    profile = {}
    for field in fields:
        series = df[field]
        keep = series.notna()
        series = series[keep]
        w = None if weights is None else np.asarray(weights)[keep.to_numpy()]

        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype="float64")
            profile[field] = {
                "kind": "numeric",
                "min": float(series.min()),
                "p1": float(series.quantile(0.01)),
                "median": (float(series.median()) if w is None
                           else _weighted_median(values, w)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
            }
        else:
            labels = series.astype(str)
            if w is None:
                mode = labels.value_counts().index[0]
            else:
                mode = pd.Series(w).groupby(labels.to_numpy()).sum().idxmax()
            profile[field] = {
                "kind": "categorical",
                "values": sorted(labels.unique().tolist()),
                "mode": mode,
            }
    return profile


def _binary_metrics(y, proba, pred, sample_weight=None):
    """The same eight numbers, computed with or without survey weights.

    Written once and called twice so the weighted and unweighted figures cannot
    drift apart -- the point of reporting both is that they are comparable.
    """
    w = sample_weight
    majority = float(np.average((y == y.mode()[0]).astype(float), weights=w))
    return {
        "roc_auc": float(roc_auc_score(y, proba, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, proba, sample_weight=w)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
        "accuracy": float(accuracy_score(y, pred, sample_weight=w)),
        "recall": float(recall_score(y, pred, sample_weight=w)),
        "precision": float(precision_score(y, pred, sample_weight=w, zero_division=0)),
        "brier_score": float(brier_score_loss(y, proba, sample_weight=w)),
        "mean_predicted_risk": float(np.average(proba, weights=w)),
        "observed_prevalence": float(np.average(y, weights=w)),
        "baseline_majority_accuracy": majority,
        "baseline_pr_auc": float(np.average(y, weights=w)),
    }


def _split(X, y, stratify=True):
    return train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if stratify else None
    )


def _scaled(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_test)


# --------------------------------------------------------------------------
# heart disease
# --------------------------------------------------------------------------

def train_heart():
    """BRFSS, the CDC's annual telephone survey of ~430,000 US adults.

    Built by ml_model/fetch_brfss.py, which does the variable mapping in the
    open. The previous training file was a pre-cleaned Kaggle derivative of the
    2015 cycle whose mapping nobody had written down, so there was no way to
    move it forward.
    """
    log.info("heart: loading BRFSS")
    source = DATA / "brfss_heart.csv"
    if not source.exists():
        raise SystemExit(
            "data/brfss_heart.csv is missing. Build it with:\n"
            "    python ml_model/fetch_brfss.py"
        )
    df = pd.read_csv(source)
    X = F.build_heart(df)
    y = df["HeartDiseaseorAttack"].astype(int)

    if "SurveyWeight" not in df.columns:
        raise SystemExit(
            "data/brfss_heart.csv has no SurveyWeight column. It predates the "
            "survey weighting; rebuild it with:\n"
            "    python ml_model/fetch_brfss.py"
        )

    # BRFSS is a stratified survey raked to census margins, so a row is not a
    # person -- it is SurveyWeight people, and that weight ranges from 0.16 to
    # 69,786. The app quotes its percentages as literal, which means they have
    # to be percentages of a population somebody belongs to.
    #
    # Where the weight is used, and where it deliberately is not:
    #
    #   evaluation   WEIGHTED. Every headline metric describes US adults. This
    #                is the whole correction: the app was reporting a 9.0%
    #                prevalence that belongs to people who answer telephone
    #                surveys, as though it described the country's 7.2%.
    #   threshold    WEIGHTED. Youden's J trades a missed case against a false
    #                alarm, and that trade should be counted per person in the
    #                population, not per person in the sample.
    #   raw_profile  Medians WEIGHTED (the "typical person" an explanation
    #                compares you against is a population claim); the p1/p99
    #                extrapolation bounds stay unweighted, because those ask
    #                what the model actually saw. See _profile().
    #   the fit      UNWEIGHTED, and this is the interesting one. Weighting a
    #                loss corrects for a sampling design that makes the sample's
    #                P(Y|X) differ from the population's. Here it does not:
    #                BRFSS rakes on age and sex, and age and sex are both
    #                features, so the model already conditions on what the
    #                design selected on. Weighting then buys no bias correction
    #                and costs effective sample size. Measured over five splits:
    #
    #                    unweighted fit   ROC-AUC 0.8524 +/- 0.0022, Brier 0.0567
    #                    weighted fit     ROC-AUC 0.8474 +/- 0.0029, Brier 0.0574
    #
    #                Both scored survey-weighted; the unweighted fit won on all
    #                five, and was the better *population*-calibrated of the two
    #                (mean predicted risk off by 0.001 against 0.003). So the
    #                weight belongs in how this model is judged and reported,
    #                not in how it is fitted. Reproduce before changing this.
    #
    # Rescaled to mean 1: relative weights unchanged, but raw BRFSS weights sum
    # to 171 million and metric code is easier to trust at a sane scale.
    weights = df["SurveyWeight"] / df["SurveyWeight"].mean()

    unweighted_prevalence = float(y.mean())
    weighted_prevalence = float(np.average(y, weights=weights))
    log.info("  %d rows, %.2f%% positive unweighted / %.2f%% weighted",
             len(y), 100 * unweighted_prevalence, 100 * weighted_prevalence)
    log.info("  weights span %.2f-%.1f (rescaled to mean 1)",
             weights.min(), weights.max())

    # Three-way split: the decision threshold is tuned on validation data so the
    # reported test metrics stay honest. The weights follow their rows by index
    # rather than being split separately, which is the only way to be sure a
    # respondent and their weight cannot come apart.
    X_fit, X_test, y_fit, y_test = _split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=0.2, random_state=SEED, stratify=y_fit
    )
    w_train = weights.loc[X_train.index].to_numpy()
    w_val = weights.loc[X_val.index].to_numpy()
    w_test = weights.loc[X_test.index].to_numpy()

    scaler = StandardScaler().fit(X_train)
    Xtr, Xval, Xte = (scaler.transform(d) for d in (X_train, X_val, X_test))

    # Gradient boosting rather than a deep forest. The previous model was a
    # 200-tree depth-20 RandomForest that serialised to 382 MB -- too large to
    # commit, and no more accurate than this.
    #
    # Deliberately NOT class_weight="balanced". Reweighting leaves ranking
    # untouched (ROC-AUC 0.8486 vs 0.8489) but wrecks calibration: mean predicted
    # risk climbs to 0.35 against a true prevalence of 0.094, and the Brier score
    # more than doubles. This page shows the user a percentage, so that
    # percentage has to mean what it says. Class imbalance is handled at the
    # decision threshold instead, which is where it belongs.
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    ).fit(Xtr, y_train)  # unweighted on purpose -- see the note above

    # Youden's J on validation data: the usual screening choice, weighting a
    # missed case and a false alarm equally. Weighted, so "equally" means per
    # person in the population rather than per person in the sample -- an
    # unweighted curve here would tune the operating point for the survey's
    # older respondents and apply it to everyone.
    val_proba = model.predict_proba(Xval)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, val_proba, sample_weight=w_val)
    threshold = float(thresholds[np.argmax(tpr - fpr)])

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, y_train)

    # Both sets, always. The weighted figures are the headline because they are
    # what the pages quote; the unweighted ones stay recorded so the size of the
    # correction is visible rather than a claim in a commit message.
    metrics = _binary_metrics(y_test, proba, pred, sample_weight=w_test)
    unweighted = _binary_metrics(y_test, proba, pred)

    metrics["baseline_note"] = (
        "Accuracy is not a meaningful headline here: predicting 'no disease' "
        "for everyone scores %.3f. Judge this model on ROC-AUC and PR-AUC. "
        "Every figure in this block is survey-weighted, so it describes US "
        "adults rather than BRFSS respondents. Predicted probabilities are "
        "calibrated against that population -- mean predicted risk tracks "
        "observed prevalence -- so the percentage shown to users is literal."
        % metrics["baseline_majority_accuracy"]
    )

    log.info("  ROC-AUC %.4f | PR-AUC %.4f (baseline %.4f) | balanced acc %.4f",
             metrics["roc_auc"], metrics["pr_auc"],
             metrics["baseline_pr_auc"], metrics["balanced_accuracy"])
    log.info("  threshold %.4f -> recall %.4f, precision %.4f",
             threshold, metrics["recall"], metrics["precision"])
    log.info("  calibration: mean predicted %.4f vs observed %.4f | Brier %.4f",
             metrics["mean_predicted_risk"], metrics["observed_prevalence"],
             metrics["brier_score"])
    log.info("  unweighted, for comparison: ROC-AUC %.4f | observed %.4f",
             unweighted["roc_auc"], unweighted["observed_prevalence"])
    log.info("  dummy accuracy %.4f vs model accuracy %.4f",
             accuracy_score(y_test, dummy.predict(Xte), sample_weight=w_test),
             metrics["accuracy"])

    return _save("heart", model, scaler, F.HEART_FEATURES, {
        "task": "binary classification",
        "target": "HeartDiseaseorAttack",
        "classes": ["No", "Yes"],
        "positive_class_index": 1,
        "decision_threshold": threshold,
        "threshold_rule": (
            "Youden's J, tuned on a held-out validation split, survey-weighted"
        ),
        "dataset": "BRFSS 2023 (CDC Behavioral Risk Factor Surveillance System)",
        "n_rows": int(len(y)),
        "estimator": type(model).__name__,
        "weighting": {
            "variable": "_LLCPWT",
            "design": "stratified, raked to census margins; no usable clustering",
            "applied_to": ["metrics", "decision_threshold", "raw_profile.median"],
            "not_applied_to": ["model.fit", "raw_profile.p1", "raw_profile.p99"],
            # What the percentages on the page are percentages *of*. The route
            # reads this rather than hardcoding a description of the comparator,
            # so retraining on a different population cannot leave the sentence
            # behind saying something that stopped being true.
            "population": "US adults",
            "represents_adults": int(df["SurveyWeight"].sum()),
            "weighted_prevalence": weighted_prevalence,
            "unweighted_prevalence": unweighted_prevalence,
            "note": (
                "Complete cases only -- respondents missing any of the 15 "
                "answers are dropped before training, and dropping is not "
                "random. These %d rows carry %.0f million adults, short of the "
                "full cycle, so read every figure here as describing US adults "
                "who answered every question."
                % (len(y), df["SurveyWeight"].sum() / 1e6)
            ),
        },
        "raw_profile": _profile(df, F.HEART_RAW, weights=df["SurveyWeight"]),
        "metrics": metrics,
        "metrics_unweighted": unweighted,
    })


# --------------------------------------------------------------------------
# migraine
# --------------------------------------------------------------------------

def train_migraine():
    log.info("migraine: loading migraine dataset")
    df = pd.read_csv(DATA / "migraine_dataset_500 (1).csv")
    df.columns = df.columns.str.strip()

    X = F.build_migraine(df)
    y = df["Migraine"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("unexpected Migraine label values")
    y = y.astype(int)

    majority = float(y.value_counts(normalize=True).max())
    log.info("  %d rows, %.1f%% positive (majority baseline %.3f)",
             len(y), 100 * y.mean(), majority)

    X_train, X_test, y_train, y_test = _split(X, y)
    scaler, Xtr, Xte = _scaled(X_train, X_test)

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        random_state=SEED,
    ).fit(Xtr, y_train)

    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "baseline_majority_accuracy": majority,
        "cv_accuracy_mean": float(
            cross_val_score(model, Xtr, y_train, cv=5, n_jobs=-1).mean()
        ),
    }
    log.info("  accuracy %.4f (baseline %.4f) | ROC-AUC %.4f | F1 %.4f",
             metrics["accuracy"], majority, metrics["roc_auc"], metrics["f1"])

    return _save("migraine", model, scaler, F.MIGRAINE_FEATURES, {
        "task": "binary classification",
        "target": "Migraine",
        "classes": ["No Migraine Risk", "Migraine Risk"],
        "positive_class_index": 1,
        "dataset": "migraine_dataset_500",
        "n_rows": int(len(y)),
        "estimator": type(model).__name__,
        "raw_profile": _profile(df, F.MIGRAINE_RAW),
        "metrics": metrics,
    })


# Two models, not four. The lifestyle score is a rubric (app/ml/lifestyle.py):
# its old training data was Gaussian noise. Sleep is an empirical lookup over
# NHANES (app/ml/sleep_risk.py): retraining on real national data showed an
# unfitted rule over two questions matched a fitted model over nine features.
TRAINERS = {
    "heart": train_heart,
    "migraine": train_migraine,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(TRAINERS), action="append",
                        help="train only these models (repeatable)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    targets = args.model or list(TRAINERS)

    summary = {}
    for name in targets:
        log.info("")
        log.info("=" * 70)
        summary[name] = TRAINERS[name]()

    log.info("")
    log.info("=" * 70)
    log.info("done: %s", ", ".join(targets))
    return summary


if __name__ == "__main__":
    main()
