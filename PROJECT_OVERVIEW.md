# LifePulse Project Overview

LifePulse is a Flask application that serves four machine-learning health screening
tools — heart disease risk, sleep disorder classification, migraine risk, and a composite
health score — plus two rule-based tools (a health calculator and a USDA nutrition
lookup). All inference runs locally; no external AI service is called.

> ⚕️ Screening tools, not a medical device. Nothing here is a diagnosis.

## Where the numbers live

Model metrics are **not** duplicated in this document. They are written to
`app/models/<name>/metadata.json` by the training script and read from there by both the
README table and the pages that display them.

```bash
python -c "import json;print(json.load(open('app/models/heart/metadata.json'))['metrics'])"
```

This used to be a table of hardcoded figures, and it disagreed with the README, which
disagreed with the HTML templates — three sources, three different accuracy claims for
the same models. A single generated source is the fix.

## Architecture

```
                      app/ml/features.py
                   (the feature contract)
                      /              \
      ml_model/train_all.py        app/ml/bundle.py
            (fits)                     (serves)
                                          |
                                    app/routes/*.py
```

Feature engineering is defined once, in `app/ml/features.py`, and imported by both the
training script and the serving layer. Routes map form fields to raw input names and
hand off — they contain no feature logic. `bundle.py` raises `FeatureContractError` if a
builder's output ever stops matching the trained artifact.

## How it works

1. **Input** — the user fills in a health form.
2. **Build** — the route maps form fields to raw names; `app/ml/features.py` engineers them.
3. **Verify** — `bundle.py` checks the columns against the trained contract and raises on mismatch.
4. **Predict** — the scaled matrix goes to the estimator; the result renders with its confidence.

## Technologies

Python 3.12 · Flask 3.1 · scikit-learn 1.8 · pandas · numpy · joblib · Bootstrap 5.3 ·
Gunicorn on Render.

## Getting started

See the [README](README.md) for installation, retraining, testing, and deployment.

## Future work

- More conditions and models
- User accounts and personalised dashboards
- Wearable-device integration for real-time tracking
- Retraining on larger, more representative datasets — the health-score model in
  particular is currently fit on synthetic data

## License

MIT.
