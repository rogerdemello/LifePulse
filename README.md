# LifePulse

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://lifepulse-9vz4.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Health screening you can take to your doctor.** Describe what you've noticed,
> get a result that shows its working, and print a summary for your appointment.
> Everything runs on this server; nothing you enter is stored.

🔗 **Live:** [lifepulse-9vz4.onrender.com](https://lifepulse-9vz4.onrender.com/)

> ⚕️ **Not a medical device.** These are screening tools built on public survey
> data. Nothing here is a diagnosis. Consult a healthcare professional.

---

## Contents

- [What makes it different](#what-makes-it-different)
- [What it does](#what-it-does)
- [Installation](#installation)
- [Models, rules and lookups](#models-rules-and-lookups)
- [Why three of six aren't models](#why-three-of-six-arent-models)
- [Datasets checked and rejected](#datasets-checked-and-rejected)
- [The safety net](#the-safety-net)
- [Explaining a result](#explaining-a-result)
- [How the ML layer is organised](#how-the-ml-layer-is-organised)
- [Design system](#design-system)
- [Testing](#testing)
- [Endpoints](#endpoints)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)

---

## What makes it different

- **You don't need to know what you want.** Describe what you noticed — "I snore and I'm tired all day" — and it routes you. Describe chest pain and it stops, and tells you to get help instead.
- **It refuses to answer when it shouldn't.** A heart rate of 0 or a BMI of 500 is rejected, not answered. A blood pressure of 190/125 interrupts with "contact a doctor" *before* anything is calculated.
- **It admits what it hasn't seen.** Where a model is used, inputs outside its training range are flagged as unreliable rather than answered confidently.
- **Every result shows its working.** Which of your answers moved the outcome, in which direction, by how much — or, where there's no model, the arithmetic itself.
- **Every result is printable.** A visit summary with your results, what drove them, and questions to ask — assembled in your browser and never uploaded.

---

## What it does

| | What it is | Basis |
|---|---|---|
| **Heart disease risk** | Gradient-boosting model | BRFSS 2023, 312k US adults |
| **Sleep screening** | Empirical lookup + criteria check | NHANES 2017-18, 4,417 US adults |
| **Migraine risk** | Gradient-boosting model | 2,000-respondent lifestyle survey |
| **Lifestyle score** | Transparent rubric | WHO / UK CMO / sleep-medicine guidance |
| **Health calculator** | Arithmetic | BMI, BMR, calorie and waist-hip formulas |
| **Nutrition lookup** | Live API + labelling thresholds | USDA FoodData Central |

Two of the six are models. That is the result of testing each one rather than
assuming — see [below](#why-three-of-six-arent-models).

---

## Installation

```bash
git clone https://github.com/rogerdemello/LifePulse.git
cd LifePulse

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then fill in SECRET_KEY

python run.py                      # http://localhost:5000
```

No Git LFS step. Both models total 0.25 MB and are committed directly, so a
plain clone gives a working app. `GET /healthz` confirms which models loaded.

### Rebuilding the data and models

Training data is gitignored — it's an input, not an output, and the BRFSS
download alone is 1.2 GB unpacked. Both fetchers are reproducible:

```bash
python ml_model/fetch_brfss.py     # -> data/brfss_heart.csv   (CDC BRFSS 2023)
python ml_model/fetch_nhanes.py    # -> data/nhanes_sleep.csv  (CDC NHANES 2017-18)
python ml_model/train_all.py       # both models
```

`data/migraine_dataset_500 (1).csv` is the one input without a documented
source — see [Known limitations](#known-limitations).

---

## Models, rules and lookups

Retrained by `python ml_model/train_all.py`. Metrics come from a held-out 20%
test split and are written to `app/models/<name>/metadata.json`, which is what
the app displays — no figure below is hardcoded anywhere in the UI.

| | Algorithm | Headline | Baseline it beats | Size |
|---|---|---|---|---|
| Heart disease | HistGradientBoosting | **ROC-AUC 0.840**, PR-AUC 0.349 | PR-AUC 0.090 (prevalence) | 0.14 MB |
| Migraine | HistGradientBoosting | **84.0%** accuracy, ROC-AUC 0.924 | 60.0% (majority class) | 0.12 MB |
| Sleep | *empirical lookup* | — | — | — |
| Lifestyle score | *rubric* | — | — | — |

### Why baselines are quoted

Only 9.0% of the heart dataset is positive, so predicting "no disease" for
everyone scores **91.0% accuracy**. A bare accuracy figure would look impressive
and mean nothing. Heart is judged on ROC-AUC and PR-AUC; everything else is
reported against the majority-class rate it has to beat.

### Calibration and the decision threshold

The heart model is deliberately trained **without** class weighting. On the
current data, rebalancing leaves ranking untouched and wrecks calibration:

| | ROC-AUC | Brier | Mean predicted risk |
|---|---|---|---|
| Unweighted (shipped) | 0.8403 | **0.069** | **0.091** |
| `class_weight="balanced"` | 0.8405 | 0.172 | 0.354 |

Observed prevalence is 0.090. The page shows the user a percentage, so that
percentage has to be literal. Class imbalance is handled at the decision
threshold instead — tuned by Youden's J on a validation split, stored as
`decision_threshold` in the metadata, and read at request time. Never a
hardcoded 0.5, which on a 9% base rate would flag almost nobody.

### Why the heart model uses BRFSS 2023

It ran on a pre-cleaned Kaggle derivative of BRFSS 2015 — a decade old and
unrefreshable, because whoever built that file did the variable mapping and
never wrote it down. `ml_model/fetch_brfss.py` does the mapping in the open
against CDC's own release and verifies it before writing.

**The newer model scores slightly lower**: ROC-AUC 0.840 against 0.848, PR-AUC
lift 3.7× against 4.0×. That is real and it is not a mapping bug — every
feature's prevalence matches the old file within a point (HighBP 0.429→0.435,
HighChol 0.424→0.424, Stroke 0.041→0.044), and the shifts that exist run the
right way (smoking 44%→39%).

It ships anyway: eight years of currency, a mapping anyone can audit, 312k
respondents instead of 254k, and an annual refresh path outweigh 0.008 ROC-AUC.
The old file's provenance could never be checked at all, which is the class of
problem the rest of this work removed.

Fruit and vegetable intake went with it. They moved ROC-AUC from 0.8485 to
0.8486, and BRFSS stopped running that module after 2015 — so two questions
buying nothing were also the only thing pinning the model to 2015. The form is
15 questions now.

---

## Why three of six aren't models

Each was tested rather than assumed, and three didn't survive.

### Lifestyle score — was a RandomForest reporting R² 0.81

- **The data was generated, not observed.** Every column in
  `synthetic_health_data.csv` is Gaussian noise around a formula with no
  clamping: 70 rows had *negative* alcohol consumption, 18 had a diet quality
  above the stated maximum of 100, and one respondent was 1.1 years old.
- **The form's encodings didn't match it.** The diet dropdown offers 1–9 while
  training saw 19.9–110.3, so selecting "9 — Excellent" put the user *below* the
  worst diet the model had ever seen. A maximally healthy profile scored 68.9/100.
- **There was nothing to predict.** "Health score" is a construct, not a
  measurable outcome, so there is no ground truth to learn.

It's a rubric now: six modifiable factors weighted against published guidance,
with every point traceable on the result page. Age is excluded deliberately —
it measures what you can change.

### Sleep — was a two-class model that couldn't say "you're fine"

Trained on a file whose respondents all had systolic 110–144 and pulse 60–89,
so the app refused to trust its own answer for anyone hypertensive — exactly the
people most likely to have sleep apnea. Retraining on NHANES 2017-18 (measured,
not self-reported, systolic 72–224) fixed the range and then showed this:

| | ROC-AUC |
|---|---|
| Snoring alone | 0.775 |
| Snoring + daytime sleepiness | **0.791** |
| …plus age, sex, BMI, blood pressure and pulse | 0.741 |
| Unfitted rule `2×snoring + sleepiness` | **0.791** |

Two questions carry the signal, extra features actively hurt, and an unfitted
rule matches a fitted model. So the page reports the survey's own numbers: of
the 418 adults who snore frequently and are often sleepy, **37% reported gasping
or stopping breathing**, against an 11.5% national average. A count, not a
prediction — nothing fitted, nothing extrapolated, no training range to fall
outside.

Insomnia is deliberately **not** predicted. Inferring it from body measurements
reaches ROC-AUC 0.616 on the same data, close enough to chance not to show
anyone. It's defined by symptoms, so those are asked directly and checked
against the standard criteria.

### Nutrition — was a hardcoded table of thirteen foods

Matched by substring, so "chicken nuggets" matched *chicken* and was described
as "low in fat", "eggplant" matched *egg*, and "oatmeal" matched nothing because
the key was "oats". Everything outside those thirteen returned an empty list.

Facts are now derived from the numbers USDA returns — UK front-of-pack
thresholds for fat, saturates, sugars and salt, US Daily Values for source
claims — so it covers the whole database and every label shows the figure behind
it.

---

## Datasets checked and rejected

Three unused CSVs sit in `data/`. None survived:

| Dataset | Why not |
|---|---|
| **Stroke** (5,110 rows, real data) | Age alone scores ROC-AUC 0.786; all six features score 0.786. The other five questions add nothing — an age lookup table wearing a form. |
| **Liver** (1,700 rows) | Good model (ROC-AUC 0.835 without a blood test) on synthetic data — every continuous column uniformly distributed, KS p > 0.05. |
| **Mental-health wearable** (10,000 rows) | 0.658 accuracy against a 0.516 baseline. Inferring a mental-health condition from heart rate and step count is not a claim this app should make. |

---

## The safety net

Three tiers, in `app/ml/safety.py`, applied before anything is shown:

| Tier | Trigger | Response |
|---|---|---|
| **Impossible** | Outside human physiology (HR 0, BMI 500, 25-hour sleep) | **400**, naming the field and its plausible range. Guessing from a typo is worse than declining — the user may act on the answer. |
| **Red flag** | Possible and urgent (BP ≥180/120, HR <40 or >120, BMI <16 or ≥40) | An interstitial *before* any result, with an explicit "show my results anyway". |
| **Out of range** | Possible, not urgent, unseen in training | The result, plus a caveat quoting the trained range. |

Tier 3's bounds come from the data — `train_all.py` records each model's actual
input ranges into `metadata.json`, so they track retraining rather than drifting
out of a hand-maintained table.

`app/ml/triage.py` adds a fourth check at the front door: `/start` scans a
free-text description for emergency presentations — chest pain, stroke signs,
severe breathlessness, thoughts of self-harm — and stops rather than routing.
Every emergency keyword is a multi-word phrase, because a single generic word
("fit" for seizure) would fire on "improve my fitness", and a warning that cries
wolf is worse than none.

> ⚠️ **The thresholds in `safety.py` and `triage.py` are clinical content.** They
> follow published guidance (AHA blood-pressure stages, standard
> bradycardia/tachycardia bounds, WHO BMI classes, NHS emergency signs), each
> cited in a comment. **None has been reviewed by a clinician.** Get that review
> before this goes in front of real users.

---

## Explaining a result

Each answer is re-scored with that one field replaced by its typical value from
training; the shift is that field's contribution. One batched prediction for the
whole explanation, no extra dependency.

```
Your self-rated general health (fair) raised your estimate by 20.0 in 100
Your high blood pressure (yes)        raised your estimate by 12.6 in 100
Your smoking history (yes)            raised your estimate by  9.8 in 100
```

It's a counterfactual, not a decomposition — contributions don't sum to the
total, because the model isn't additive. Result pages lead with plain language
and keep the ROC-AUC, Brier score and threshold behind a "technical version"
disclosure.

---

## How the ML layer is organised

Feature engineering lives in exactly one place and both sides import it:

```
                  app/ml/features.py
               (the feature contract)
                  /              \
  ml_model/train_all.py      app/ml/bundle.py
        (fits)                   (serves)
                                     |
                               app/routes/*.py
```

Routes map form fields to raw names and hand off; they contain no feature logic.
If a builder's output stops matching the trained contract, `bundle.py` raises
`FeatureContractError` rather than predicting — and
`tests/test_feature_contract.py` fails first.

This structure exists for a reason. Each route used to reimplement its model's
feature engineering inline, and the loader filled in `0.0` for any name it could
not find. The names drifted, nothing raised, and every model was served vectors
far outside its training distribution — sleep was receiving
`Blood_Pressure_Mean` at **z = −16.7**.

---

## Design system

Colour carries meaning: red means seek care, amber means worth raising, green
means reassuring, everything else is neutral. The stylesheet previously put a
gradient on every card, which meant the genuinely urgent panels had to compete
with decoration.

`app/static/css/style.css` is token-driven, and those tokens also drive
Bootstrap's own `--bs-*` variables — overriding only ours left Bootstrap's
components on its built-in light palette, which made card titles and form labels
invisible in dark mode. Contrast ratios are computed from the stylesheet in
`tests/test_design_system.py`, so a palette change fails a test rather than
showing up in a screenshot months later.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

**325 tests** (324 pass; one skips when the gitignored data is absent), covering:

- **Feature contract** — builder output matches each trained artifact exactly, in order
- **Fail-fast** — a missing or unrecognised input raises instead of defaulting to zero
- **Training/serving parity** — a CSV row and the equivalent form dict produce identical vectors
- **Model quality** — each model beats its baseline; heart stays calibrated; the heart data stays recent
- **Safety** — each tier fires correctly, including the BP 190/125 and HR 0 cases that once returned a calm "No Sleep Disorder"
- **Triage** — emergency phrasings are caught, including inflections like "ending my life"; ordinary ones like "improve my fitness" never trigger a false alarm
- **Sleep & lifestyle** — the lookup table is monotonic and the rubric's components sum to its total
- **Explanations** — directions aren't inverted below 50% risk; age actually moves the heart prediction
- **Nutrition** — derived facts match published thresholds; search ranks relevance above brevity (the USDA API is mocked, so CI stays hermetic)
- **Front end** — every referenced asset exists, pages render without JavaScript, landmarks and skip links are present, contrast clears WCAG AA in both themes
- **Rate limiting** — bursts are throttled per client, GETs never are
- **Observability** — every request carries an id reaching the logs and the error page, and request logs never contain the answers

Tests needing the gitignored CSVs skip cleanly when the data isn't present.

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Homepage |
| `/start` | GET/POST | Symptom-led entry point, with emergency detection |
| `/heart_disease/` | GET/POST | Heart disease risk |
| `/sleep/` | GET/POST | Sleep apnea and insomnia screening |
| `/migraine/` | GET/POST | Migraine risk |
| `/health-score/` | GET/POST | Lifestyle score |
| `/health/` | GET | Health calculator form |
| `/health/result` | POST | Calculator results |
| `/health/calculate_metrics` | POST | Calculator as JSON |
| `/nutrition/` | GET/POST | USDA nutrition lookup |
| `/summary` | GET | Printable visit summary (client-side only) |
| `/privacy` | GET | What is and isn't stored |
| `/healthz` | GET | Liveness probe; reports which models loaded |

---

## Project structure

```
LifePulse/
├── app/
│   ├── app.py                  # application factory
│   ├── observability.py        # request ids, timing, optional Sentry
│   ├── ratelimit.py            # per-client throttle on the model endpoints
│   ├── ml/
│   │   ├── features.py         # THE feature contract — training and serving
│   │   ├── bundle.py           # model loading, contract enforcement, explanations
│   │   ├── safety.py           # physiological limits, red flags, range caveats
│   │   ├── triage.py           # symptom routing + emergency detection
│   │   ├── sleep_risk.py       # NHANES lookup + insomnia criteria
│   │   ├── lifestyle.py        # the scoring rubric
│   │   └── guidance.py         # questions to ask your doctor
│   ├── models/<name>/          # model.joblib, scaler.joblib,
│   │                           # features.json, metadata.json
│   ├── routes/                 # start, heart, sleep, migraine, health_score,
│   │                           # calculator_routes, nutrition, support
│   ├── templates/              # Jinja templates + partials
│   ├── static/                 # css, js (steps, summary, charts, toast)
│   └── utils/
│       ├── calculator.py       # BMI / BMR / calorie rules
│       ├── nutrition.py        # USDA FoodData client
│       └── nutrition_facts.py  # labelling thresholds and portions
├── ml_model/
│   ├── fetch_brfss.py          # CDC BRFSS -> data/brfss_heart.csv
│   ├── fetch_nhanes.py         # CDC NHANES -> data/nhanes_sleep.csv
│   └── train_all.py            # retrains both models
├── tests/                      # 325 tests
├── data/                       # training inputs (gitignored)
├── .github/workflows/ci.yml    # pytest + boot check + contract check
├── requirements.txt            # pinned runtime deps
├── runtime.txt                 # Python 3.12 for Render
├── Procfile                    # gunicorn wsgi:app
└── wsgi.py                     # WSGI entry point
```

### A note on CSRF

There is deliberately no CSRF protection. Every form is unauthenticated and
changes no state — no account, no database, no stored result — so a forged
cross-site submission would achieve nothing beyond making a stranger's browser
compute a score it never displays. Adding tokens would mean a session cookie and
a dependency to defend against nothing.

**This stops being true the moment accounts or saved history exist.** If that
changes, CSRF protection goes in at the same time, not after.

---

## Deployment

**Render:**

1. Connect the repository
2. Build: `pip install -r requirements.txt`
3. Start: `gunicorn wsgi:app`
4. Set `SECRET_KEY` — the app refuses to start in production without it
5. Optionally set `USDA_API_KEY` to enable `/nutrition/`, and `SENTRY_DSN` (plus
   `pip install sentry-sdk`) for error monitoring
6. Point the health check at `/healthz`

`runtime.txt` pins Python 3.12 to match the pinned wheels. No Git LFS needed.

**Locally, production-style:**

```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

---

## Known limitations

Stated plainly, because the rest of this README argues for taking limitations
seriously.

- **The clinical thresholds have not been reviewed by a clinician.** This is the
  gating item before real use.
- **Migraine's data provenance is unverifiable.** `migraine_dataset_500 (1).csv`
  has 2,000 rows despite "500" in the filename and no documented source. Its
  distributions look real — strongly non-uniform, with genuine missing values,
  unlike the synthetic files that were rejected — but that isn't the same as
  knowing where it came from. Every other input can name its source.
- **BRFSS is self-reported.** "Have you ever been told you have high blood
  pressure" is not a measurement.
- **No keyboard or screen-reader testing.** The structural work is done —
  landmarks, skip link, `aria-current`, focus rings, reduced-motion — but nobody
  has driven the app with a screen reader.
- **Single language, mixed guidance.** UK front-of-pack thresholds sit beside US
  Daily Values on the same nutrition page.

---

## License

MIT — see [LICENSE](LICENSE).

## Developer

**Roger Demello** · [GitHub](https://github.com/rogerdemello) · [LinkedIn](https://www.linkedin.com/in/roger-demello)

## Acknowledgments

- CDC BRFSS 2023 and NHANES 2017-18 (public domain)
- USDA FoodData Central
- WHO, UK Chief Medical Officers and AASM/SRS guidance underpinning the rubric
- Bootstrap and Bootstrap Icons
