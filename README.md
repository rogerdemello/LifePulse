# 🏥 LifePulse - AI-Powered Health Analytics Platform

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://lifepulse-9vz4.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Health screening you can take to your doctor.** Machine learning models for heart disease, sleep disorders, migraine risk, and lifestyle scoring — each result explained, caveated where the model is out of its depth, and printable for an appointment. All inference runs locally; nothing is stored.

🔗 **Live Application:** [https://lifepulse-9vz4.onrender.com/](https://lifepulse-9vz4.onrender.com/)

> ⚕️ **Not a medical device.** These are screening tools trained on public survey data. Nothing here is a diagnosis. Consult a healthcare professional.

### What makes it different

- **You don't need to know what you want.** Describe what you've noticed — "I snore and I'm tired all day" — and it routes you. Describe chest pain and it stops, and tells you to get help instead.

- **It refuses to answer when it shouldn't.** A heart rate of 0 or a BMI of 500 is rejected, not answered. A blood pressure of 190/125 interrupts with "contact a doctor" *before* any model runs.
- **It admits what it hasn't seen.** Where a model is used, inputs outside its training range are flagged as unreliable rather than answered confidently.
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

### 😴 Sleep Screening
Apnea signs scored against **real observed rates** from NHANES 2017-18 (4,417 US
adults), and an insomnia check against the standard diagnostic criteria. **Not a
model** — the percentage shown is a count from the survey, so there is no training
range to fall outside.

### 🤕 Migraine Risk Assessment
Binary risk from 10 lifestyle inputs plus 10 derived interactions (sleep×stress,
water/caffeine, screen/sleep).

### 🧮 Lifestyle Score
A 0–100 checklist over six modifiable factors, scored against WHO activity and BMI
guidance, sleep-medicine consensus and UK alcohol limits. **Not a model** — the
result page shows exactly where every point came from. Age is deliberately
excluded: it measures what you can change.

### 🥗 Nutrition Lookup
Any food in USDA FoodData Central, with what its numbers mean: UK front-of-pack
thresholds for fat, saturates, sugars and salt, and US Daily Values for
"good/excellent source" claims. Every label shows the figure it came from.

### 🧮 Health Calculator
Rule-based BMI, BMR, calorie and waist-hip calculations. No model.

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

The two models are retrained from scratch by `python ml_model/train_all.py`. Metrics come
from a held-out 20% test split and are written to `app/models/<name>/metadata.json`,
which is what the app displays — no figure in this table is hardcoded anywhere in the UI.

| Feature | Algorithm | Headline metric | Baseline it beats | Size |
|---|---|---|---|---|
| Heart Disease | HistGradientBoosting | **ROC-AUC 0.849**, PR-AUC 0.368 | PR-AUC 0.094 (prevalence) | 0.16 MB |
| Sleep | *empirical lookup, not a model* | — | — | — |
| Migraine | HistGradientBoosting | **84.0%** accuracy, ROC-AUC 0.924 | 60.0% (majority class) | 0.12 MB |
| Lifestyle Score | *rubric, not a model* | — | — | — |

### Why baselines are quoted

Only 9.4% of the heart-disease dataset is positive, so predicting "no disease" for
everyone scores **90.6% accuracy**. A bare accuracy figure would look impressive and mean
nothing. Heart is therefore judged on ROC-AUC and PR-AUC, and the other models are
reported against the majority-class rate they have to beat.

### Why the lifestyle score isn't a model

It was one — a RandomForest reporting R² 0.81. Three things were wrong with it:

- **The data was generated, not observed.** Every column in `synthetic_health_data.csv`
  is Gaussian noise around a formula, with no clamping: 70 rows had *negative* alcohol
  consumption, 18 had a diet quality above the stated maximum of 100, and one
  respondent was 1.1 years old. The R² measured how well it recovered a random
  number generator.
- **The form's encodings didn't match it.** The diet dropdown offers 1–9 while
  training saw 19.9–110.3, so selecting "9 — Excellent" put the user *below* the
  worst diet the model had ever seen. A maximally healthy profile scored 68.9/100.
- **There was nothing to predict.** "Health score" is a construct, not a measurable
  outcome, so there is no ground truth a model could learn.

A rubric wins on every axis that matters here: explainable by construction, every
weight a stated judgement traceable to public guidance, and it cannot silently
drift from the form. The weights are editorial and the page says so.

### Why sleep isn't a model either

The old sleep model was trained on a file whose respondents all had a systolic
blood pressure between 110 and 144 and a resting pulse between 60 and 89, so the
app refused to trust its own answer for anyone hypertensive — exactly the people
most likely to have sleep apnea.

Retraining on **NHANES 2017-18** (a real, public-domain US national survey with
*measured* blood pressure and pulse, 4,417 adults, systolic 72–224) fixed the
range and then showed something more useful:

| | ROC-AUC |
|---|---|
| snoring alone | 0.775 |
| snoring + daytime sleepiness | **0.791** |
| …plus age, sex, BMI, blood pressure and pulse | 0.741 |
| unfitted rule `2×snoring + sleepiness` | **0.791** |

Two questions carry the signal, extra features actively hurt, and an unfitted
rule matches a fitted model. So the page reports the survey's own numbers: of the
418 adults who snore frequently and are often sleepy, 37% reported gasping or
stopping breathing in their sleep, against a 11.5% national average. It is a
count, not a prediction — nothing fitted, nothing extrapolated.

Insomnia is deliberately **not** predicted. On the same data, inferring it from
body measurements reaches ROC-AUC 0.616, close enough to chance not to show
anyone. It is defined by symptoms, so those are asked about directly and checked
against the standard criteria.

Rebuild the table: `python ml_model/fetch_nhanes.py`

### Datasets I checked and rejected

Three unused CSVs sit in `data/`. All three failed:

| Dataset | Why not |
|---|---|
| **Stroke** (5,110 rows, real) | Age alone scores ROC-AUC 0.786; all six features score 0.786. The other five questions add nothing — it would be an age lookup table wearing a form. |
| **Liver** (1,700 rows) | Good model (ROC-AUC 0.835 without a blood test) on synthetic data — every continuous column is uniformly distributed, KS p > 0.05. |
| **Mental health wearable** (10,000 rows) | 0.658 accuracy against a 0.516 baseline. Inferring a mental-health condition from heart rate and step count is not something this app should claim to do. |

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

No Git LFS step. The two models total ~0.5 MB and are committed directly, so a plain
clone gives you a working app. (`GET /healthz` confirms which models loaded.)

### Retraining

The training CSVs are gitignored — they are inputs, not outputs, and BRFSS alone is
22 MB. Place them in `data/` and run:

```bash
python ml_model/train_all.py                  # all three
python ml_model/train_all.py --model heart    # just one
```

| File | Source |
|---|---|
| `heart_disease_health_indicators_BRFSS2015.csv` | BRFSS 2015 (Kaggle) |
| `Sleep_health_and_lifestyle_dataset (1).csv` | Sleep Health and Lifestyle (Kaggle) |
| `migraine_dataset_500 (1).csv` | Migraine triggers dataset |


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
names drifted, nothing raised, and every model was served vectors far outside their
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

256 tests covering:

- **Feature contract** — builder output matches each trained artifact exactly, in order
- **Fail-fast** — a missing or unrecognised input raises instead of defaulting to zero
- **Training/serving parity** — a CSV row and the equivalent form dict produce identical vectors
- **Model quality** — each model beats its baseline; heart stays calibrated; sleep predicts all three classes
- **Safety** — each tier fires correctly, including the BP 190/125 and HR 0 cases that once returned a calm "No Sleep Disorder"
- **Explanations** — directions aren't inverted below 50% risk; age actually moves the heart prediction
- **Front end** — every referenced asset exists, pages render without JavaScript, landmarks and skip links are present
- **Rate limiting** — bursts are throttled per client, GETs never are
- **Triage** — emergency phrasings are caught (including inflections like "ending my life"), and ordinary ones like "improve my fitness" never trigger a false alarm
- **Multi-step forms** — every field stays in the served markup, so the form works with JavaScript off
- **Nutrition** — derived facts match the published thresholds, search ranks relevance above brevity, upstream failures degrade readably (the USDA API is mocked, so CI stays hermetic)
- **Routes** — every page renders, bad input returns 400 without leaking a traceback

Tests that need the gitignored CSVs skip cleanly when the data isn't present.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Homepage |
| `/start` | GET/POST | Symptom-led entry point |
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
