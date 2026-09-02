# LifePulse

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://lifepulse-9vz4.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Health screening you can take to your doctor.** Describe what you've noticed,
> get a result that shows its working, and print a summary for your appointment.
> Every model runs on this server; nothing you enter is stored on it.

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
- [The emergency check, measured](#the-emergency-check-measured)
- [The safety net](#the-safety-net)
- [Explaining a result](#explaining-a-result)
- [How the ML layer is organised](#how-the-ml-layer-is-organised)
- [Azure OpenAI](#azure-openai)
- [Design system](#design-system)
- [Continuous integration](#continuous-integration)
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
| Heart disease | HistGradientBoosting | **ROC-AUC 0.855**, PR-AUC 0.321 | PR-AUC 0.073 (prevalence) | 0.14 MB |
| Migraine | HistGradientBoosting | **84.0%** accuracy, ROC-AUC 0.924 | 60.0% (majority class) | 0.12 MB |
| Sleep | *empirical lookup* | — | — | — |
| Lifestyle score | *rubric* | — | — | — |

### Why baselines are quoted

Only 7.3% of the held-out test split is positive once it is weighted to US
adults, so predicting "no disease" for everyone scores **92.7% accuracy**. A bare
accuracy figure would look impressive and mean nothing. Heart is judged on
ROC-AUC and PR-AUC; everything else is reported against the majority-class rate
it has to beat.

### Survey weights, and what they are and aren't for

BRFSS is a stratified survey raked to census margins. A row is not a person —
it is `_LLCPWT` people, and that weight spans 0.16 to 69,786. Ignoring it is not
a rounding error:

| Across all 312,166 rows | Prevalence |
|---|---|
| Unweighted — people BRFSS reached | 0.090 |
| **Weighted — US adults** | **0.072** |

The app tells people its percentages are literal, so they have to be percentages
of a population somebody belongs to. It was quoting the first number as though it
were the second: a 25% overstatement of how common heart disease is, used as the
comparator on every result.

Where the weight is applied is a separate question from whether it is used at
all, and the answer is not "everywhere":

- **Evaluation, the threshold, and the "typical person"** an explanation compares
  you against are all weighted. Each is a claim about a population.
- **The extrapolation bounds** (`p1`/`p99` in `raw_profile`) stay unweighted.
  Those ask what the model actually saw, and a survey weight doesn't change what
  was in the training rows.
- **The fit is unweighted**, which is the one worth explaining. Weighting a loss
  corrects for a design that makes the sample's `P(Y|X)` differ from the
  population's. Here it doesn't: BRFSS rakes on age and sex, and age and sex are
  both features, so the model already conditions on what the design selected on.
  Weighting then buys no correction and costs effective sample size. Over five
  splits, all scored survey-weighted:

  | | ROC-AUC | Brier | Calibration gap |
  |---|---|---|---|
  | **Unweighted fit (shipped)** | **0.8524 ± 0.0022** | **0.0567** | **−0.0011** |
  | Weighted fit | 0.8474 ± 0.0029 | 0.0574 | −0.0029 |

  The unweighted fit won on every split and was the better *population*-calibrated
  of the two. So the weight belongs in how this model is judged, not in how it is
  fitted — and the estimator that ships is byte-for-byte the one that shipped
  before this work. What changed is that it is now scored, thresholded and
  described against the country rather than against the survey.

`_PSU` is deliberately not carried. It looks like a cluster identifier, but ids
are numbered within stratum and every `(_STSTR, _PSU)` pair in the cycle is
unique — one record per cluster, so there is nothing for a "cluster-aware" split
to group by.

### Who it works less well for

A single ROC-AUC of 0.855 is an average over 62,434 test rows, and averages hide
their tails. `metadata.json` carries weighted metrics split six ways — sex, age
band, sex × age, race/ethnicity, income and education — and the result page tells
each reader how the model does for *their* sex and age band, not just overall.

Two findings the aggregate was hiding:

| | n | Observed | Predicted | O/E | ROC-AUC |
|---|---|---|---|---|---|
| Age 35–49 | 11,903 | 0.024 | 0.024 | 1.01 | 0.863 |
| Age 50–64 | 17,231 | 0.082 | 0.075 | **1.10** | 0.805 |
| Age 65–79 | 19,821 | 0.146 | 0.151 | 0.96 | 0.772 |
| **Age 80+** | 5,718 | 0.207 | 0.223 | 0.93 | **0.686** |
| **Asian** | 1,611 | 0.028 | 0.039 | **0.71** | 0.897 |
| White | 47,690 | 0.081 | 0.079 | 1.02 | 0.847 |

- **Discrimination falls off with age.** 0.863 at 35–49 against 0.686 past 80 —
  and past 80 is where a reader is most likely to act on the answer. The model
  ranks the oldest group barely better than a coin weighted by prevalence.
- **It over-states risk for Asian adults by about 40%** — predicting 3.9% where
  2.8% had the outcome. Ranking within the group is fine (0.897, the best of any
  group); it is the level that is wrong.

Race, income and education are **evaluation strata only**. The model is never
given them as inputs — feeding race into a clinical risk score is the mistake
behind a generation of race-adjusted equations now being withdrawn, and
`tests/test_model_quality.py` fails if one ever reaches the feature contract.
They are carried to answer the question "who does this work worse for", which
cannot be answered without them.

Two tests hold the floor: no group above n=500 may fall outside an
observed/predicted band of 0.6–1.4, and none may drop below ROC-AUC 0.65. Those
are "something has broken" bounds, not a claim that the current numbers are good.
A third asserts the 80+ band is still the weakest, so if a retrain fixes it, the
test fails and this section gets updated with it.

### Calibration and the decision threshold

The heart model is deliberately trained **without** class weighting. On the
current data, rebalancing leaves ranking untouched and wrecks calibration:

| | ROC-AUC | Brier | Mean predicted risk |
|---|---|---|---|
| No class weighting (shipped) | 0.8551 | **0.057** | **0.073** |
| `class_weight="balanced"` | 0.8506 | 0.136 | 0.291 |

Observed prevalence is 0.073. The page shows the user a percentage, so that
percentage has to be literal — and mean predicted risk lands within 0.0004 of it.
Class imbalance is handled at the decision threshold instead — tuned by Youden's
J on a weighted validation split, stored as `decision_threshold` in the metadata,
and read at request time. Never a hardcoded 0.5, which on a 7% base rate would
flag almost nobody.

### Why the heart model uses BRFSS 2023

It ran on a pre-cleaned Kaggle derivative of BRFSS 2015 — a decade old and
unrefreshable, because whoever built that file did the variable mapping and
never wrote it down. `ml_model/fetch_brfss.py` does the mapping in the open
against CDC's own release and verifies it before writing.

**Compared like for like the newer model scores slightly lower**: unweighted
ROC-AUC 0.840 against the 2015 file's 0.848, PR-AUC lift 3.7× against 4.0×. That
is real and it is not a mapping bug — every feature's prevalence matches the old
file within a point (HighBP 0.429→0.435, HighChol 0.424→0.424, Stroke
0.041→0.044), and the shifts that exist run the right way (smoking 44%→39%).

It ships anyway: eight years of currency, a mapping anyone can audit, 312k
respondents instead of 254k, and an annual refresh path outweigh 0.008 ROC-AUC.
The old file's provenance could never be checked at all, which is the class of
problem the rest of this work removed. It also carries the survey weights, which
the Kaggle derivative had dropped — so the 2015 file could never have been
scored against the population at all.

The headline 0.855 above is the same model scored *survey-weighted*. Both numbers
are in `metadata.json`, under `metrics` and `metrics_unweighted`, so the size of
the correction stays visible instead of becoming a claim in a commit message.

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

## The emergency check, measured

`check_emergency` in `app/ml/triage.py` is the highest-stakes path in the
product — it's what stands between somebody typing "crushing chest pain" and a
questionnaire. For its whole life it was validated by fifteen hand-picked
examples, all of them cases somebody had already thought of.

`tests/data/emergency_phrasings.csv` is 110 labelled phrasings built to include
the ones nobody had: negations, third-person reports, and ordinary complaints
that happen to contain an emergency phrase's words.

| | Sensitivity | False alarms |
|---|---|---|
| Bag-of-stems (before) | 95.6% | **46.2%** |
| **Now** | **100%** | **10.8%** |

Nearly half of ordinary complaints used to fire the stop sign. "no chest pain",
"my dad had chest pain last year, am I at risk", and "back pain and a chest
infection" all did. That isn't cosmetic: a warning that goes off on the wrong
sentences is one people learn to click past, and it blocks the person asking
about their father's heart attack from the heart assessment they came for.

Two rules fixed it, both suppressing only on unambiguous, local evidence:

- **Negation** — a cue in the few words before or inside the match, unless the
  cue belongs to the keyword itself, or "I don't want to live" would negate its
  own phrase. It looks *backwards only*: "chest pressure that won't go away" and
  "better off dead without me" both carry a cue after the symptom, and
  suppressing on those cost five real emergencies when it was tried.
- **Attribution** — a third-party subject before the match with no first-person
  word in between. "my dad had chest pain" attributes; "my dad worries, but I
  have chest pain" doesn't.

**Proximity was tried and removed**, which is the interesting one. Requiring a
keyword's words to sit close together fixes the remaining false alarms — and
breaks the case this module exists for. Someone types "I get a lot of pain when
I walk upstairs", is asked where, and answers "it is in my chest"; those two
words land eight apart in the joined text, and that is precisely when the stop
sign matters most. On this data proximity was *anti-correlated* with
correctness: the false alarm had the words closer together than the real
emergency did. So the incidental matches are left to fire, in the safe
direction.

Two bugs surfaced on the way, both invisible without the measurement:

- `_normalise` replaced apostrophes with a space, so "don't" became `don` + `t`.
  Every negation cue a person actually types — don't, haven't, can't, won't —
  was split in half, so the cue the matcher looked for never appeared.
- "can" is a three-letter prefix of "cant", so prefix matching had "I can
  breathe fine" matching the keyword "cant breathe" — telling somebody with a
  snoring complaint to call an ambulance.

The two floors are asymmetric on purpose. Sensitivity must be 100%; the
false-alarm bound is a ceiling on drift, not a target.

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

## Azure OpenAI

Optional, off by default, and scoped so that `/privacy` stays true as written.

**At runtime it sees one thing: what is typed into the "what's bothering you"
box on `/start`**, so it can route "my chest feels tight climbing stairs" where
a keyword matcher misses. Nothing from any assessment form is sent — not the
answers, not the results, not the scores.

The routing is conversational: the agent may ask up to two clarifying
questions before it commits, because "I'm exhausted all the time" is three of
the six assessments and one question about snoring settles it. Answers to those
questions are sent the same way, so it is the sentences typed into that box —
`/privacy` says so in those words.

Five rules the routing never breaks:

1. **Emergency detection is not in that path.** It runs first, on local keyword
   rules. Whether someone is told to call an ambulance must not depend on a
   third party being reachable.
2. **It runs on every turn, over everything typed so far** — not over the newest
   message. Emergency keywords are multi-word phrases matched against the words
   present, so "I get pain walking upstairs" then "it is in my chest" is a
   cardiac flag that neither turn raises alone.
3. **The model may only pick from the fixed concern set.** Its answer is looked
   up in a table and discarded if it isn't a real key, so it can route but never
   invent a destination.
4. **The question budget is enforced by the server**, not by the prompt. After
   two, the reply is read only for a destination.
5. **Any failure falls back to keywords.** Unconfigured, slow, rate-limited or
   wrong are ordinary conditions, not error pages. So is a blank question, a
   400-character one, and a hallucinated key.

The exchange is never stored. It rides in hidden fields and comes back with the
next post — the same trick the red-flag interstitial uses to replay a pending
submission without a session — so it arrives attacker-controlled and is
validated for role, emptiness and length before anything is assembled from it.

**Better result copy is generated at build time, not per request.** The shape of
every sentence is knowable in advance — there are only so many (field,
direction) pairs and result bands — so `ml_model/generate_phrasings.py` walks
that space once on a developer's machine, screens the output for anything that
instructs or falsely reassures, and commits it as `app/ml/phrasings.json`. At
runtime the app picks a sentence and fills in the person's numbers locally.

That's the trade: the copy can't react to an individual, and in exchange no
assessment data ever leaves the server. `phrasings.json` is optional — without
it every page keeps its built-in wording.

```bash
python ml_model/generate_phrasings.py --dry-run   # show exactly what would be sent
python ml_model/generate_phrasings.py             # regenerate the copy
python ml_model/generate_phrasings.py --check     # verify the committed file
```

The client is a thin wrapper over the REST API using `requests` — already a
dependency — rather than the SDK, so every outbound byte is assembled in one
function that a test can assert on. `tests/test_azure_openai.py` checks the
request body is entirely determined by fixed app text plus the typed sentence,
and that the assessment routes never reach the network at all.

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_DEPLOYMENT` to enable it; `/healthz` reports whether it's on.

---

## Design system

Colour carries meaning: red means seek care, amber means worth raising, green
means reassuring, everything else is neutral. The stylesheet previously put a
gradient on every card, which meant the genuinely urgent panels had to compete
with decoration.

`app/static/css/style.css` is token-driven, and contrast ratios are computed
from it in `tests/test_design_system.py`, so a palette change fails a test rather
than showing up in a screenshot months later.

### Nothing loads from a CDN

The page used to pull Bootstrap's CSS, its icon font and Inter from jsDelivr and
Google Fonts. `style.css` restyled `.card` and `.btn` on top of that but never
defined `.container`, `.row`, `.col-md-6` or a single spacing utility — which the
templates use **145 times** between them. A blocked CDN therefore didn't degrade
the styling, it collapsed every page into one unstyled column, and 448 tests
couldn't see it because catching it needs a network fault.

| | Before | Now |
|---|---|---|
| Third-party requests per page | 4 | **0** |
| CSS shipped | ~330 kB (Bootstrap + icon font CSS) | **62 kB** (`layout.css` + `style.css`) |
| Web fonts | Inter + Bootstrap Icons | none; system stack |
| Icons | 2,000-glyph font | the 60 used, as an inline SVG sprite |

- `app/static/css/layout.css` — the grid, spacing and utility classes the
  templates actually use. A subset, deliberately, and not a Bootstrap clone.
- `app/templates/_icons.html` — built by `python tools/build_icon_sprite.py`
  from whatever the templates, the JavaScript and `app/ml/triage.py` reference,
  so an icon can't be added to a page and missing from the sprite.

Three tests hold it: every class the templates use must have a rule somewhere in
this repository, no template may reference another origin, and the sprite must
contain exactly the icons referenced — no more and no fewer. CI checks the last
one again against the *rendered* pages, which is what a browser is served.

### Security headers

There were none. `app/security.py` adds them, and dropping the CDNs is what makes
the useful one possible:

```
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-…'; …
Referrer-Policy: no-referrer
X-Content-Type-Options: nosniff
Permissions-Policy: camera=(), microphone=(), geolocation=(), …
Strict-Transport-Security: max-age=31536000  (HTTPS only)
```

`script-src` carries a per-response nonce and **no** `'unsafe-inline'`, so an
injected `<script>` doesn't run even if escaping fails somewhere — worth having
because `/start` echoes back a sentence the visitor typed. `Referrer-Policy:
no-referrer` matters more than usual here: the assessment paths alone say what
someone was worried about.

`style-src` still allows inline styles. That's a stated gap rather than an
oversight — a nonce can't cover style *attributes*, and there are about thirty
of them plus five per-page `<style>` blocks. An injected style can restyle a
page but cannot execute, which is why scripts were the half to close first.

---

## Continuous integration

`.github/workflows/ci.yml`, on every push to `main` and every pull request:

| Gate | What it catches |
|---|---|
| `ruff check .` | Its first run found a variable left behind by an earlier change, five imports orphaned when two models were deleted, and two `zip()` calls that would truncate silently rather than raise. Config in `ruff.toml`. |
| `pytest --cov-fail-under=85` | Currently 87%. The floor sits below the suite on purpose — it catches a change that quietly stops being tested, not every refactor. |
| Boot check | The app starts and every model loads. |
| Feature-contract check | The committed artifacts still match `app/ml/features.py`. |
| Same-origin check | The **rendered** pages fetch nothing from anywhere else. |
| `pip-audit` | Known vulnerabilities in the pinned runtime deps. Its first run found seven across five packages, all since bumped. Runs as its own job: a newly disclosed CVE shouldn't block a PR that didn't touch dependencies, but somebody should be told. |

`.github/dependabot.yml` raises weekly pip updates and monthly Actions updates.
scikit-learn, numpy, pandas, scipy and joblib are excluded from *automatic* PRs —
not from updates, but because merging one without running
`python ml_model/train_all.py` leaves artifacts that may no longer load. Bump
those by hand and commit the retrained models with them.

`Dockerfile` builds on the same Python 3.12 as CI and the deploy, runs as a
non-root user, and deliberately bakes in no `SECRET_KEY` — `app/app.py` refuses
to start in production without one, and an image carrying a key would make every
deployment from it share a signing key. `tests/test_deploy.py` asserts both.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

**477 tests** (476 pass; one skips when the gitignored data is absent), covering:

- **Feature contract** — builder output matches each trained artifact exactly, in order
- **Fail-fast** — a missing or unrecognised input raises instead of defaulting to zero
- **Training/serving parity** — a CSV row and the equivalent form dict produce identical vectors
- **Model quality** — each model beats its baseline; heart stays calibrated; the heart
  data stays recent; the survey weights stay applied and the correction stays visible;
  no subgroup drifts outside its calibration or discrimination floor, and race, income
  and education never reach the feature contract
- **Safety** — each tier fires correctly, including the BP 190/125 and HR 0 cases that once returned a calm "No Sleep Disorder"
- **Triage** — measured against 110 labelled phrasings, not sampled: every labelled
  emergency is caught, and negated ("no chest pain") and third-person ("my dad had
  chest pain") reports no longer fire the stop sign
- **Sleep & lifestyle** — the lookup table is monotonic and the rubric's components sum to its total
- **Explanations** — directions aren't inverted below 50% risk; age actually moves the heart prediction
- **Nutrition** — derived facts match published thresholds; search ranks relevance above brevity (the USDA API is mocked, so CI stays hermetic)
- **Front end** — every referenced asset exists, pages render without JavaScript, landmarks and skip links are present, contrast clears WCAG AA in both themes
- **Rate limiting** — bursts are throttled per client, GETs never are
- **Security headers** — every response carries them, including the error pages;
  the CSP nonce changes per response and the markup uses the one it was given;
  HSTS is sent over HTTPS and never over plain HTTP
- **Front end** — every class the templates use has a rule in this repository,
  no template references another origin, and the icon sprite holds exactly the
  icons referenced
- **Observability** — every request carries an id reaching the logs and the error page, and request logs never contain the answers
- **Azure OpenAI** — the outbound body contains only fixed app text plus the typed sentence; assessment routes never reach the network; emergencies are detected before any call; every failure mode falls back to keywords; the privacy page changes when it's switched on

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
│   ├── security.py             # CSP, HSTS, Referrer-Policy, Permissions-Policy
│   ├── azure_openai.py         # optional LLM client; only /start uses it
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
│   ├── templates/              # Jinja templates, partials, icon sprite
│   ├── static/                 # css (layout + style), js (steps, summary,
│   │                           # charts, toast) — all served from this origin
│   └── utils/
│       ├── calculator.py       # BMI / BMR / calorie rules
│       ├── nutrition.py        # USDA FoodData client
│       └── nutrition_facts.py  # labelling thresholds and portions
├── ml_model/
│   ├── fetch_brfss.py          # CDC BRFSS -> data/brfss_heart.csv
│   ├── fetch_nhanes.py         # CDC NHANES -> data/nhanes_sleep.csv
│   └── train_all.py            # retrains both models
├── tools/
│   └── build_icon_sprite.py    # rebuilds app/templates/_icons.html
├── tests/                      # 476 tests
├── data/                       # training inputs (gitignored)
├── .github/
│   ├── workflows/ci.yml        # lint, tests + coverage floor, boot check,
│   │                           # contract check, same-origin check, pip-audit
│   └── dependabot.yml          # weekly pip, monthly actions
├── Dockerfile                  # same Python as CI and the deploy
├── ruff.toml                   # lint config, chosen to find bugs not taste
├── requirements.txt            # pinned runtime deps
├── LICENSE                     # MIT
├── .python-version             # Python 3.12 for Render
├── Procfile                    # gunicorn wsgi:app
└── wsgi.py                     # WSGI entry point
```

### A note on CSRF

There is deliberately no CSRF protection. Every form is unauthenticated and
changes no server-side state — no account, no database, nothing written to disk
— so a forged cross-site submission would achieve nothing beyond making a
stranger's browser compute a score it never displays. Adding tokens would mean a
session cookie and a dependency to defend against nothing.

The visit summary does persist, in the visitor's own `localStorage`, which is
worth checking against this reasoning rather than waving through. It doesn't
change the conclusion: a forged POST can render a result page, but writing to
the summary needs a click on **Add to visit summary** on a page served from this
origin, and an attacker's page cannot read or write another origin's
`localStorage`. The forged request still achieves nothing.

**This stops being true the moment accounts or server-side history exist.** If
that changes, CSRF protection goes in at the same time, not after.

---

## Deployment

**Render:**

1. Connect the repository
2. Build: `pip install -r requirements.txt`
3. Start: `gunicorn wsgi:app`
4. Set `SECRET_KEY` — the app refuses to start in production without it
5. Optionally set `USDA_API_KEY` for `/nutrition/`, `SENTRY_DSN` (plus
   `pip install sentry-sdk`) for error monitoring, and the `AZURE_OPENAI_*`
   variables for language-model routing on `/start`
6. Point the health check at `/healthz`

`.python-version` pins Python 3.12 to match the pinned wheels, and it has to be
that file: Render reads a `PYTHON_VERSION` environment variable or
`.python-version`, and ignores the `runtime.txt` this repository used to carry.
Without it Render falls back to a default that depends on when the service was
created, and numpy 1.26 and pandas 2.1 publish no wheels past 3.12 — so the
build does not run slower, it fails compiling pandas from source. If a
`PYTHON_VERSION` variable is set in the Render dashboard it wins over this
file; make sure it says 3.12 or remove it. No Git LFS needed.

If you enable Sentry, note what `app/observability.py` turns off and why. Error
monitoring is the most likely route for health answers to leave the server by
accident, because the defaults are built for apps whose request bodies aren't
sensitive: `max_request_body_size="never"` withholds the submitted form, and
`include_local_variables=False` withholds it a second time from the traceback,
where the form dict would otherwise sit in a stack frame. `tests/test_observability.py` asserts the arguments reaching `sentry_sdk.init`, so a
future change that drops one fails rather than silently starts uploading
answers.

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
- **No screen reader has been listened to.** The app has been driven by keyboard
  through every page and every result state, and audited against the browser's
  accessibility tree — which is what NVDA and VoiceOver read from, and which
  found 41 unlabelled fields, missing `<h1>`s and a skip link that did nothing.
  So the names, roles and levels are right. Whether it *sounds* right read
  aloud — whether the questions are phrased to be heard rather than seen — is
  not something a tree dump can answer, and nobody has checked.
- **The small-screen menu needs JavaScript.** Below 992px the navigation is
  behind a toggle, and without script it cannot be opened. Everything else
  works without JavaScript, including the whole triage conversation; the footer
  links and the homepage's own list of assessments give a way around it. A
  `<details>` would fix it, but a closed `<details>` cannot be forced open by
  CSS at the desktop breakpoint, and the checkbox-and-label alternative is
  announced as a checkbox rather than a disclosure.
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
