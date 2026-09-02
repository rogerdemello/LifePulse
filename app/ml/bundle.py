"""Model loading and inference.

Replaces the old ``app/utils/onnx_inference.py``, which despite its name never
touched ONNX -- it loaded joblib pickles, and built its input matrix with
``input_data.get(f, 0)``. Any feature the caller failed to supply under the
exact trained name silently became 0.0, which the scaler then mapped to an
extreme z-score. Nothing ever raised.

Here the feature matrix is built by ``app.ml.features``, the same code the
training scripts use, and a mismatch raises ``FeatureContractError``.

Artifact layout, per model, under ``app/models/<name>/``::

    model.joblib      fitted estimator
    scaler.joblib     fitted StandardScaler
    features.json     ordered feature names -- the contract
    metadata.json     metrics, class labels, library versions, train date

``features.json`` and ``metadata.json`` are JSON rather than pickles on
purpose: the feature list is exactly what you still need to be able to read
when the pickles stop loading against a newer scikit-learn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.features import (
    BUILDERS,
    FEATURES,
    FeatureContractError,
    describe_value,
    label_for,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Factor:
    """One answer's measured contribution to a result."""

    field: str
    label: str
    value: str
    typical: str
    delta: float

    @property
    def direction(self):
        return "raised" if self.delta > 0 else "lowered"

    @property
    def magnitude(self):
        return abs(self.delta)

    @property
    def note(self):
        """A pre-written sentence for this factor, if one was generated.

        Comes from app/ml/phrasings.json, built offline. Returns None when the
        file is absent, which is the default -- the page then shows only the
        measured contribution, exactly as before.
        """
        from app.ml.phrasings import explanation_for

        return explanation_for(self.field, self.direction)



MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_NAMES = ("heart", "migraine")


class ModelNotAvailable(RuntimeError):
    """Raised when a model's artifacts are absent or unreadable."""


class ModelBundle:
    """An estimator plus everything needed to feed it correctly."""

    def __init__(self, name, model, scaler, features, metadata):
        self.name = name
        self.model = model
        self.scaler = scaler
        self.features = list(features)
        self.metadata = metadata or {}
        self.classes = self.metadata.get("classes")
        self._build = BUILDERS[name]

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, name):
        if name not in BUILDERS:
            raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
        directory = MODELS_DIR / name
        try:
            features = json.loads((directory / "features.json").read_text("utf-8"))
            metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
            model = joblib.load(directory / "model.joblib")
            scaler = joblib.load(directory / "scaler.joblib")
        except FileNotFoundError as exc:
            raise ModelNotAvailable(
                f"{name}: missing artifact {exc.filename}. "
                f"Run `python ml_model/train_all.py --model {name}` to build it."
            ) from exc
        except Exception as exc:
            raise ModelNotAvailable(f"{name}: could not load artifacts: {exc}") from exc

        expected = FEATURES[name]
        if list(features) != list(expected):
            raise ModelNotAvailable(
                f"{name}: the saved model was trained on a different feature set than "
                f"app.ml.features defines. Retrain it. "
                f"saved={len(features)} cols, code={len(expected)} cols"
            )
        return cls(name, model, scaler, features, metadata)

    # -- inference --------------------------------------------------------

    def _matrix(self, raw):
        """Build, verify, and scale the feature matrix for ``raw``."""
        frame = self._build(raw)
        if list(frame.columns) != self.features:
            saved = set(self.features)
            built = set(frame.columns)
            raise FeatureContractError(
                self.name,
                missing=sorted(saved - built),
                unexpected=sorted(built - saved),
                detail="builder output does not match the trained contract",
            )
        # Hand the scaler the DataFrame, not a bare array: it was fitted with
        # feature names, so this makes scikit-learn re-check the column names
        # and order on every call -- a second, independent guard on the contract.
        return self.scaler.transform(frame)

    def predict(self, raw):
        return self.model.predict(self._matrix(raw))

    def predict_proba(self, raw):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.name} is a regressor; use predict()")
        return self.model.predict_proba(self._matrix(raw))

    def predict_one(self, raw):
        """Predict for a single record and return a scalar."""
        return self.predict(raw)[0]

    def predict_label(self, raw):
        """Predict a single record and map the code to its class label."""
        code = int(self.predict(raw)[0])
        if self.classes is None:
            return code
        return self.classes[code]

    def proba_one(self, raw):
        """Return ``{class_label: probability}`` for a single record."""
        probs = self.predict_proba(raw)[0]
        labels = self.classes or list(range(len(probs)))
        # strict=True: if metadata.json ever lists a different number of classes
        # than the estimator predicts, zip would silently drop the extras and
        # the page would label a probability with the wrong outcome. Better to
        # raise, which prediction_errors turns into an error page.
        return dict(zip(labels, (float(p) for p in probs), strict=True))

    # -- explanation ------------------------------------------------------

    def _outcome(self, matrix, class_index=None):
        """Reduce a scaled matrix to one comparable number per row."""
        if class_index is None:
            return np.asarray(self.model.predict(matrix), dtype=float)
        return self.model.predict_proba(matrix)[:, class_index]

    def explain(self, raw, top=4):
        """Which of the user's answers moved this result, and by how much.

        For each answer, re-predict with that one field replaced by its typical
        value from training (median for numbers, mode for categories) and
        measure the shift. The difference is that field's contribution.

        This is a counterfactual, not a decomposition: contributions will not
        sum to the total, because the model is not additive. It answers the
        question a person actually asks -- "what about *me* made this come out
        this way?" -- which is what makes a result worth discussing with a
        doctor rather than just reading.

        Every variant is built and scored in a single batched call, so the whole
        explanation costs roughly one extra prediction.
        """
        profile = self.metadata.get("raw_profile") or {}
        fields = [f for f in profile if f in raw]
        if not fields:
            return []

        rows = [dict(raw)]
        for field_name in fields:
            stats = profile[field_name]
            typical = stats["median"] if stats["kind"] == "numeric" else stats["mode"]
            rows.append({**raw, field_name: typical})

        frame = self._build(pd.DataFrame(rows))
        matrix = self.scaler.transform(frame)

        class_index = None
        if hasattr(self.model, "predict_proba"):
            # For a binary model, always explain the *positive* class -- that is
            # the number on the page. Taking the argmax instead inverts every
            # direction whenever the risk lands below 50%: a 45.6% risk would
            # report "your high blood pressure lowered it", because it was
            # silently explaining the probability of *not* having the disease.
            class_index = self.metadata.get("positive_class_index")
            if class_index is None:
                # Multiclass: explain the class the user was actually shown.
                class_index = int(np.argmax(self.model.predict_proba(matrix[:1])[0]))

        scores = self._outcome(matrix, class_index)
        actual, counterfactuals = scores[0], scores[1:]
        scale = 100.0 if class_index is not None else 1.0

        factors = []
        # strict=True for the same reason: one counterfactual was scored per
        # field, so a length mismatch means the batch came back wrong and every
        # contribution after the gap would be attributed to the wrong answer.
        for field_name, alternative in zip(fields, counterfactuals, strict=True):
            delta = (actual - alternative) * scale
            if abs(delta) < 0.05:
                continue  # below the rounding the page displays; noise
            stats = profile[field_name]
            typical = stats["median"] if stats["kind"] == "numeric" else stats["mode"]
            factors.append(Factor(
                field=field_name,
                label=label_for(field_name),
                value=describe_value(field_name, raw[field_name], self.name),
                typical=describe_value(field_name, typical, self.name),
                delta=float(delta),
            ))

        factors.sort(key=lambda f: abs(f.delta), reverse=True)
        return factors[:top]

    def __repr__(self):
        return f"<ModelBundle {self.name}: {len(self.features)} features>"


# --------------------------------------------------------------------------
# process-wide cache
# --------------------------------------------------------------------------

_cache = {}


def get_model(name):
    """Return the cached bundle for ``name``, loading it on first use."""
    if name not in _cache:
        _cache[name] = ModelBundle.load(name)
        log.info("loaded %s model (%d features)", name, len(_cache[name].features))
    return _cache[name]


def try_get_model(name):
    """Return the bundle, or ``None`` if its artifacts are unavailable.

    Routes use this so a missing model degrades to a clear service message
    instead of taking down app startup.
    """
    try:
        return get_model(name)
    except (ModelNotAvailable, ValueError) as exc:
        log.warning("%s model unavailable: %s", name, exc)
        return None


def clear_cache():
    """Drop cached bundles. Used by tests after retraining."""
    _cache.clear()
    _metadata_cache.clear()


# --------------------------------------------------------------------------
# metadata, without loading the estimators
# --------------------------------------------------------------------------

_metadata_cache = {}


def load_metadata(name):
    """Read one model's ``metadata.json``. Returns ``{}`` if it is absent.

    Cheap enough to call from a template context processor -- it never touches
    the pickles. Templates use this instead of hardcoding accuracy figures,
    which is how the app ended up advertising 51% migraine accuracy on one page
    and 82% in the README for the same model.
    """
    if name not in _metadata_cache:
        path = MODELS_DIR / name / "metadata.json"
        try:
            _metadata_cache[name] = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            _metadata_cache[name] = {}
    return _metadata_cache[name]


def all_metadata():
    """``{model_name: metadata}`` for every model, for template rendering."""
    return {name: load_metadata(name) for name in MODEL_NAMES}
