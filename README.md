# 🏥 LifePulse - AI-Powered Health Analytics Platform

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://lifepulse-9vz4.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Health screening you can take to your doctor.** Machine learning models for heart disease, sleep disorders, migraine risk, and lifestyle scoring — each result explained, caveated where the model is out of its depth, and printable for an appointment. All inference runs locally; nothing is stored.

🔗 **Live Application:** [https://lifepulse-9vz4.onrender.com/](https://lifepulse-9vz4.onrender.com/)

> ⚕️ **Not a medical device.** These are screening tools trained on public survey data. Nothing here is a diagnosis. Consult a healthcare professional.

### What makes it different

- **It refuses to answer when it shouldn't.** A heart rate of 0 or a BMI of 500 is rejected, not answered. A blood pressure of 190/125 interrupts with "contact a doctor" *before* any model runs.
- **It admits what it hasn't seen.** The sleep model was trained on people with systolic 110–144 and resting heart rates of 60–89. Outside that it says so, rather than returning a confident guess.
- **Every result is explained.** Which of your answers moved the outcome, in which direction, by how much.
- **Every result is printable.** A visit summary with your results, what drove them, and questions to ask — assembled in your browser and never uploaded.

---

## 📋 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Models](#-ml-models)
- [Installation](#-installation)
- [How the ML layer is organised](#-how-the-ml-layer-is-organised)
- [Testing](#-testing)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [License](#-license)

---

## ✨ Features

### ❤️ Heart Disease Risk
Cardiovascular risk from 17 survey answers, expanded to 25 engineered features.
Returns a **calibrated probability** — mean predicted risk (9.4%) matches observed
prevalence (9.4%) — rather than a score out of 100.

### 😴 Sleep Disorder Screening
Three-way classification: **no disorder**, insomnia, or sleep apnea. Blood pressure
is entered as real numbers and parsed, so any reading works.

### 🤕 Migraine Risk Assessment
Binary risk from 10 lifestyle inputs plus 10 derived interactions (sleep×stress,
water/caffeine, screen/sleep).

### 🧮 Health Score
A 0–100 composite from seven lifestyle inputs. **The training data for this one is
synthetic**, so treat the score as illustrative.

### 🧮 Health Calculator & 🥗 Nutrition Lookup
Rule-based BMI/BMR/calorie/waist-hip calculations, and food lookup via the USDA
FoodData Central API. Neither uses a model.

---

## 🛠️ Tech Stack

**Backend:** Flask 3.1 · Gunicorn · scikit-learn 1.8 · pandas · numpy · joblib
**Frontend:** Bootstrap 5.3 · Bootstrap Icons 1.11 · AOS 2.3 · custom CSS
**Deployment:** Render · Python 3.12

Every dependency is pinned in `requirements.txt`. The scientific stack in particular
must match the versions the models in `app/models/` were built with — bump the pins and
rerun `python ml_model/train_all.py` together, never separately.

---

## 🤖 ML Models

All four are retrained from scratch by `python ml_model/train_all.py`. Metrics come
from a held-out 20% test split and are written to `app/models/<name>/metadata.json`,
which is what the app displays — no figure in this table is hardcoded anywhere in the UI.

| Feature | Algorithm | Headline metric | Baseline it beats | Size |
|---|---|---|---|---|
| Heart Disease | HistGradientBoosting | **ROC-AUC 0.849**, PR-AUC 0.368 | PR-AUC 0.094 (prevalence) | 0.16 MB |
| Sleep Disorder | HistGradientBoosting | **85.8%** accuracy, 79.6% balanced | 70.0% (majority class) | 0.24 MB |
| Migraine | HistGradientBoosting | **84.0%** accuracy, ROC-AUC 0.924 | 60.0% (majority class) | 0.12 MB |
| Health Score | RidgeCV | **R² 0.812** | R² 0.809 (linear on raw features) | 0.8 KB |

### Why baselines are quoted

Only 9.4% of the heart-disease dataset is positive, so predicting "no disease" for
everyone scores **90.6% accuracy**. A bare accuracy figure would look impressive and mean
nothing. Heart is therefore judged on ROC-AUC and PR-AUC, and the other models are
reported against the majority-class rate they have to beat.

### Calibration and the decision threshold

The heart model is deliberately trained **without** class weighting. Rebalancing leaves
ranking untouched (ROC-AUC 0.8486 vs 0.8489) but inflates mean predicted risk from 0.094
to 0.35 and more than doubles the Brier score. Since the page shows the user a
percentage, that percentage has to be literal. Class imbalance is handled at the decision
threshold instead — tuned by Youden's J on a validation split, stored as
`decision_threshold` in the model's metadata, and read at request time (**never** a
hardcoded 0.5, which on a 9.4% base rate would flag almost nobody).

---

## 🚀 Installation

```bash
git clone https://github.com/rogerdemello/LifePulse.git
cd LifePulse

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then fill in SECRET_KEY

python run.py                      # http://localhost:5000
```

No Git LFS step. The four models total ~0.5 MB and are committed directly, so a plain
clone gives you a working app. (`GET /healthz` confirms which models loaded.)

### Retraining

The training CSVs are gitignored — they are inputs, not outputs, and BRFSS alone is
22 MB. Place them in `data/` and run:

```bash
python ml_model/train_all.py                  # all four
python ml_model/train_all.py --model heart    # just one
```

| File | Source |
|---|---|
| `heart_disease_health_indicators_BRFSS2015.csv` | BRFSS 2015 (Kaggle) |
| `Sleep_health_and_lifestyle_dataset (1).csv` | Sleep Health and Lifestyle (Kaggle) |
| `migraine_dataset_500 (1).csv` | Migraine triggers dataset |
| `synthetic_health_data.csv` | Generated locally |

---

## 🧩 How the ML layer is organised

Feature engineering lives in exactly one place, `app/ml/features.py`, and both sides
import it:

```
                      app/ml/features.py
                   (the feature contract)
                      /              \
                     /                \
      ml_model/train_all.py        app/ml/bundle.py
            (fits)                     (serves)
                                          |
                                    app/routes/*.py
```

Routes map form fields to raw names and hand off; they contain no feature logic. If a
builder's output ever stops matching the trained contract, `bundle.py` raises
`FeatureContractError` rather than predicting — and `tests/test_feature_contract.py`
fails first.

This structure exists for a reason. Each route used to reimplement its model's feature
engineering inline, and the loader filled in `0.0` for any name it could not find. The
names drifted, nothing raised, and all four models were served vectors far outside their
training distributions — the sleep model was receiving `Blood_Pressure_Mean` at
**z = −16.7**. Adding a feature now means editing one file, and the tests will tell you
if the artifacts need retraining.

---

## 🛟 The safety net

Three tiers, in `app/ml/safety.py`, applied before anything is shown:

| Tier | Trigger | Response |
|---|---|---|
| **Impossible** | Outside human physiology (HR 0, BMI 500, 25-hour sleep) | **400**, naming the field and its plausible range. Guessing from a typo is worse than declining — the user may act on the answer. |
| **Red flag** | Possible and urgent (BP ≥180/120, HR <40 or >120, BMI <16 or ≥40) | An interstitial *before* the model runs, with an explicit "show my results anyway". |
| **Out of range** | Possible, not urgent, but unseen in training | The result, plus a visible caveat quoting the trained range. |

Tier 3's bounds come from the data — `train_all.py` records each model's actual
input ranges into `metadata.json`, so they track retraining rather than drifting
out of a hand-maintained table.

> ⚠️ The tier-2 thresholds follow published guidance (AHA blood-pressure stages,
> standard bradycardia/tachycardia bounds, WHO BMI classes), each cited in a
> comment. **They have not been reviewed by a clinician.** Get that review before
> promoting this as a pre-appointment tool.

## 🔍 Explaining a result

Each answer is re-scored with that one field replaced by its typical value from
training; the shift is that field's contribution. No extra dependency, and one
batched prediction for the whole explanation.

```
Your self-rated general health (fair) raised your estimated risk by 20.0 points
Your high blood pressure (yes)        raised your estimated risk by 12.6 points
Your smoking history (yes)            raised your estimated risk by  9.8 points
```

It's a counterfactual, not a decomposition — contributions don't sum to the
total, because the model isn't additive. It answers the question people actually
ask, which is what makes a result worth discussing rather than just reading.

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
pytest
```

161 tests covering:

- **Feature contract** — builder output matches each trained artifact exactly, in order
- **Fail-fast** — a missing or unrecognised input raises instead of defaulting to zero
- **Training/serving parity** — a CSV row and the equivalent form dict produce identical vectors
- **Model quality** — each model beats its baseline; heart stays calibrated; sleep predicts all three classes
- **Safety** — each tier fires correctly, including the BP 190/125 and HR 0 cases that once returned a calm "No Sleep Disorder"
- **Explanations** — directions aren't inverted below 50% risk; age actually moves the heart prediction
- **Front end** — every referenced asset exists, pages render without JavaScript, landmarks and skip links are present
- **Rate limiting** — bursts are throttled per client, GETs never are
- **Routes** — every page renders, bad input returns 400 without leaking a traceback

Tests that need the gitignored CSVs skip cleanly when the data isn't present.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Homepage |
| `/heart_disease/` | GET/POST | Heart disease risk |
| `/sleep/` | GET/POST | Sleep disorder screening |
| `/migraine/` | GET/POST | Migraine risk |
| `/health-score/` | GET/POST | Composite health score |
| `/health/` | GET | Health calculator form |
| `/health/result` | POST | Calculator results |
| `/health/calculate_metrics` | POST | Calculator as JSON |
| `/nutrition/` | GET/POST | USDA nutrition lookup |
| `/healthz` | GET | Liveness probe; reports which models loaded |

---

## 📁 Project Structure

```
LifePulse/
├── app/
│   ├── app.py                  # application factory
│   ├── ratelimit.py            # per-client throttle on the model endpoints
│   ├── ml/
│   │   ├── features.py         # THE feature contract — training and serving
│   │   ├── bundle.py           # model loading, contract enforcement, explanations
│   │   ├── safety.py           # physiological limits, red flags, range caveats
│   │   └── guidance.py         # questions to ask your doctor
│   ├── models/<name>/          # model.joblib, scaler.joblib,
│   │                           # features.json, metadata.json
│   ├── routes/
│   │   ├── support.py          # form collection, validation, error handling
│   │   ├── heart.py  sleep.py  migraine.py  health_score.py
│   │   ├── calculator_routes.py
│   │   └── nutrition.py
│   ├── templates/              # Jinja templates (_caveats, _explanation,
│   │                           # _save_summary, urgent, summary, privacy)
│   ├── static/                 # CSS, JS, images
│   └── utils/
│       ├── calculator.py       # BMI / BMR / calorie rules
│       └── nutrition.py        # USDA FoodData client
├── ml_model/train_all.py       # retrains every model
├── tests/                      # contract, parity, quality, safety, front-end
├── data/                       # training CSVs (gitignored)
├── .github/workflows/ci.yml    # pytest + boot check + contract check
├── requirements.txt            # pinned runtime deps
├── runtime.txt                 # Python 3.12 for Render
├── Procfile                    # gunicorn wsgi:app
└── wsgi.py                     # WSGI entry point
```

### A note on CSRF

There is deliberately no CSRF protection. Every form is unauthenticated and
changes no state — there is no account, no database, and no stored result, so a
forged cross-site submission would achieve nothing beyond making a stranger's
browser compute a health score it never displays. Adding tokens would mean
introducing a session cookie and a dependency to defend against nothing.

**This stops being true the moment accounts or saved history exist.** If that
changes, CSRF protection goes in at the same time, not after.

---

## 🌍 Deployment

**Render:**

1. Connect the repository
2. Build: `pip install -r requirements.txt`
3. Start: `gunicorn wsgi:app`
4. Set `SECRET_KEY` (required — the app refuses to start in production without it)
5. Optionally set `USDA_API_KEY` to enable `/nutrition/`

`runtime.txt` pins Python 3.12 to match the pinned wheels. No Git LFS configuration
needed. Point the health check at `/healthz`.

**Locally, production-style:**
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 👨‍💻 Developer

**Roger Demello**
- GitHub: [@rogerdemello](https://github.com/rogerdemello)
- LinkedIn: [Connect](https://www.linkedin.com/in/roger-demello)
- Live Demo: [LifePulse](https://lifepulse-9vz4.onrender.com/)

---

## 🙏 Acknowledgments

- BRFSS 2015 dataset (heart disease indicators)
- Sleep Health & Lifestyle dataset
- USDA FoodData Central
- Bootstrap and AOS
