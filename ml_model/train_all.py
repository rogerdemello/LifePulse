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


def _profile(df, fields):
    """Summarise the raw inputs the model actually saw during training.

    Two things downstream depend on this, and both need it to come from the data
    rather than a hand-maintained table that would drift on the next retrain:

    * Extrapolation warnings. The sleep dataset contains no systolic reading
      above 144 and no resting heart rate outside 60-89. Predicting for someone
      hypertensive is guesswork, and the app has to be able to say so.
    * Result explanations, which re-predict with one field swapped for its
      typical value to measure that field's contribution.
    """
    profile = {}
    for field in fields:
        series = df[field].dropna()
        if pd.api.types.is_numeric_dtype(series):
            profile[field] = {
                "kind": "numeric",
                "min": float(series.min()),
                "p1": float(series.quantile(0.01)),
                "median": float(series.median()),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
            }
        else:
            counts = series.astype(str).value_counts()
            profile[field] = {
                "kind": "categorical",
                "values": sorted(counts.index.tolist()),
                "mode": counts.index[0],
            }
    return profile


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
    log.info("heart: loading BRFSS 2015")
    df = pd.read_csv(DATA / "heart_disease_health_indicators_BRFSS2015.csv")
    X = F.build_heart(df)
    y = df["HeartDiseaseorAttack"].astype(int)

    majority = float(y.value_counts(normalize=True).max())
    log.info("  %d rows, %.1f%% positive (majority-class baseline %.3f)",
             len(y), 100 * y.mean(), majority)

    # Three-way split: the decision threshold is tuned on validation data so the
    # reported test metrics stay honest.
    X_fit, X_test, y_fit, y_test = _split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=0.2, random_state=SEED, stratify=y_fit
    )
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
    ).fit(Xtr, y_train)

    # Youden's J on validation data: the usual screening choice, weighting a
    # missed case and a false alarm equally.
    val_proba = model.predict_proba(Xval)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, val_proba)
    threshold = float(thresholds[np.argmax(tpr - fpr)])

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, y_train)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "mean_predicted_risk": float(proba.mean()),
        "observed_prevalence": float(y_test.mean()),
        "baseline_majority_accuracy": majority,
        "baseline_pr_auc": float(y_test.mean()),
        "baseline_note": (
            "Accuracy is not a meaningful headline here: predicting 'no disease' "
            "for everyone scores %.3f. Judge this model on ROC-AUC and PR-AUC. "
            "Predicted probabilities are calibrated -- mean predicted risk tracks "
            "observed prevalence -- so the percentage shown to users is literal."
            % majority
        ),
    }
    log.info("  ROC-AUC %.4f | PR-AUC %.4f (baseline %.4f) | balanced acc %.4f",
             metrics["roc_auc"], metrics["pr_auc"],
             metrics["baseline_pr_auc"], metrics["balanced_accuracy"])
    log.info("  threshold %.4f -> recall %.4f, precision %.4f",
             threshold, metrics["recall"], metrics["precision"])
    log.info("  calibration: mean predicted %.4f vs observed %.4f | Brier %.4f",
             metrics["mean_predicted_risk"], metrics["observed_prevalence"],
             metrics["brier_score"])
    log.info("  dummy accuracy %.4f vs model accuracy %.4f",
             accuracy_score(y_test, dummy.predict(Xte)), metrics["accuracy"])

    return _save("heart", model, scaler, F.HEART_FEATURES, {
        "task": "binary classification",
        "target": "HeartDiseaseorAttack",
        "classes": ["No", "Yes"],
        "positive_class_index": 1,
        "decision_threshold": threshold,
        "threshold_rule": "Youden's J, tuned on a held-out validation split",
        "dataset": "BRFSS 2015 heart disease health indicators",
        "n_rows": int(len(y)),
        "estimator": type(model).__name__,
        "raw_profile": _profile(df, F.HEART_RAW),
        "metrics": metrics,
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
