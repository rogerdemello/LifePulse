# LifePulse — what's left

Written 3 September 2026, after Phase 0 and Phase 1 landed. Phases and item
numbers follow the original advancement plan so the earlier commits still line
up with it.

---

## Where things stand

**Updated 3 September 2026, mid-session.** PR #8 merged to `main` before this
note was written — the section below is stale, kept for the history.

~~Merged to `main`: survey weighting (#6) and subgroup performance (#7).~~
~~Open as [PR #8](https://github.com/rogerdemello/LifePulse/pull/8): the~~
~~self-hosted front end, CI gates, the measured emergency check, migraine~~
~~provenance, and the risk interval.~~

Since then: `main`'s CI was red from the moment PR #8 landed —
`tests/data/emergency_phrasings.csv` was gitignored by an unanchored `data/`
rule and had never actually been committed, so every clean checkout failed
`test_triage.py`. Fixed in
**[PR #11](https://github.com/rogerdemello/LifePulse/pull/11)** (open).

486 tests, 87% coverage, ruff clean, `pip-audit` clean, zero third-party
requests per page.

**Before doing anything that touches a model**, rebuild the training extract —
the committed CSV predates three columns added in #7:

```bash
python ml_model/fetch_brfss.py
python ml_model/train_all.py
```

`ml_model/train_all.py` exits with that instruction rather than silently
training without them, so you'll be told if you forget.

---

## Start here tomorrow

Three things, in this order. The first two are small and clear the desk; the
third has a long lead time and should be set running before anything else
competes for attention.

**Status, mid-session:** #1 mostly done (see below), #2 decided and (b)
implemented, #3 still not started — it's external and nobody here can do it.

### 1. Merge or close out PR #8 — 30 min

~~It's ready. One thing in it is genuinely unverified: the Dockerfile has never
been built. Docker wasn't available where it was written.~~ PR #8 merged
before this session started. The Dockerfile claim is still unverified — Docker
isn't available in this environment either. Still open:

```bash
docker build -t lifepulse .
docker run --rm -p 5000:5000 -e SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')" lifepulse
curl -s localhost:5000/healthz
```

If it builds and `/healthz` returns 200 with both models loaded, the claim in
the README's deployment section stops needing its caveat. Needs a machine with
Docker.

### 2. Decide two things only you can decide — 20 min

Both are written up honestly in the README and neither is a defect. They need a
judgement, not more work:

**a. How strict should the subgroup guards be?** `tests/test_model_quality.py`
currently fails a build only outside observed/predicted 0.6–1.4 and below
ROC-AUC 0.65. The two known weak spots both pass:

| | n | Observed | Predicted | O/E | ROC-AUC |
|---|---|---|---|---|---|
| Age 80+ | 5,718 | 0.207 | 0.223 | 0.93 | **0.686** |
| Asian | 1,611 | 0.028 | 0.039 | **0.71** | 0.897 |

They're named in the README rather than tuned away. The alternative is a build
that refuses to ship a model that over-states one group's risk by 40%. That's a
one-line change and a defensible position — it's your call which you want.

**Decided: leave as-is.** Keep shipping with both named honestly in the
README; revisit if a future subgroup comes in worse.

**b. Does the 80+ band deserve different treatment on the page?** ROC-AUC 0.686
is barely better than ranking by prevalence, and past 80 is where a reader is
most likely to act on the answer. The page already tells them their band's
figures. The stronger options are to widen their interval, or to decline to give
a number and route them to a doctor — which is what `app/ml/safety.py` already
does for inputs the model has no evidence for. Worth a decision before Phase 2
builds more on top of it.

**Decided: widen the interval, not route to a doctor.** 80+ isn't *no*
evidence — it's weaker evidence, and the model still beats prevalence-ranking
there, so `safety.py`'s "no evidence" pattern was a mismatch. Implemented:
`risk_bins_80plus` in `ml_model/train_all.py`, computed the same way as
`risk_bins` but restricted to the 80+ test cohort, so every band comes out
2–7× wider — a real sample-size effect, not a fudge factor. `app/routes/heart.py`
prefers it for an 80+ reader. One thin bin (0–2%, n=6) gets dropped by a new
`min_n` floor rather than reported, and falls back to the general table.
Tests in `test_model_quality.py` and `test_explanations.py`; README's "Who it
works less well for" section updated. Committed on `fix/80plus-risk-interval`,
not yet pushed/PR'd — waiting on the go-ahead.

### 3. Start the clinician conversation — item 3.1, external

The long pole. Everything else can proceed in parallel; this can't be
compressed once started, so start it first. **Not started — this needs you,
not an agent:** find and engage a clinician reviewer. See Phase 3.1 below for
the scope to hand them once you have one.

---

## Phase 2 — give people a reason to come back

Everything so far makes the product defensible. This is what makes it *used*.
The constraint that shapes all of it: **no account, no server-side storage,
nothing that makes the privacy page start hedging.**

### 2.1 Local-first history and trends — 1.5 weeks

`static/js/summary.js` is one-shot: take an assessment, print it, done. A
screening tool's value compounds over repetitions — "my blood pressure has
climbed three readings running" is worth more to a doctor than any single
result — and the architecture already supports it, because results are assembled
client-side and never posted back.

- Dated history in `localStorage`, capped and prunable, with a visible list the
  person controls.
- Sparklines on each result page: this reading against the last four.
- Export/import as a single JSON file, so a phone and a laptop can share a
  history without a server ever holding one.
- One-tap clear, and a plain statement of where the data lives.

**Watch:** the README's CSRF reasoning is explicitly conditional on there being
no server-side state. This work doesn't change that — but the moment anything
persists server-side, CSRF protection goes in the same commit, as the README
already commits to.

**Done when** a returning user sees their trend, the privacy page needs no new
caveat, and clearing the history is one click from anywhere.

### 2.2 Peer-group comparators — 3 days · *unblocked by #6*

~~"Compared with 7.2% among US adults" still tells a 28-year-old almost
nothing. The comparator that means something is same sex, same age band —
and #7 already computes exactly those cells.~~

~~Weighted prevalence by sex × age band, stored at training time, read at
request time. No new model, no new dependency; the same pattern
`app/ml/sleep_risk.py` already uses, where a table beat a model.~~

**Done for heart — [PR #13](https://github.com/rogerdemello/LifePulse/pull/13)
(open).** The headline now reads "Across US adults, it's about 7 in 100 —
among men aged 50–64, it's about 10 in 100, from 8,132 survey respondents."
No new metadata: reused the same `subgroup` value the calibration line
already builds from `subgroups.sex_age`, so both go quiet together on a cell
too small to trust.

**Not done for sleep or migraine**, and not blocked so much as genuinely
different work: migraine has no verified demographic breakdown to comparator
against (its data provenance is already flagged as unreliable — the right
fix there is 2.3's ID-Migraine swap, not more derived statistics on shaky
data), and sleep's NHANES table is keyed on two symptom questions, not sex
and age. Either would need new data work, not a read of data already on
hand. So "every risk figure" from the original done-criterion below is now
one of two applicable models, not four.

~~**Done when** every risk figure sits beside the rate for people of the same
sex and age band, with the cell size stated.~~

### 2.3 Breadth through validated instruments — 1 week each

The instinct to add conditions is right; adding them as models trained on found
CSVs is what produced the two assessments this project already had to delete.
Published instruments arrive with peer-reviewed sensitivity and specificity you
can print on the page — a stronger claim than any model here can make.

- **ID-Migraine** — three questions. This is the *real* fix for the migraine
  model, which #8 only marked. Doing this completes the pattern already applied
  to sleep and the lifestyle score: when the data can't support a model, use the
  published rule.
- **FINDRISC** — type 2 diabetes, eight questions, validated across multiple
  European cohorts.
- **AUDIT-C** — alcohol, three questions, the standard primary-care screen.
- **PHQ-2 / GAD-2** — depression and anxiety. These need a mandatory
  crisis-resource interstitial wired into the existing red-flag machinery
  **before** they ship, not after.
- **STOP-BANG** — could sharpen the NHANES apnea table, which deliberately uses
  only two questions.

**Done when** each new assessment cites its validation study on its own result
page, with the same rigour the sleep table cites NHANES.

### 2.4 Heard, not just parsed — 4 days

The README still says no screen reader has been listened to. The accessibility
tree says the names and roles are right. It cannot say whether a result reads
aloud as a sentence a frightened person can follow.

- One NVDA pass and one VoiceOver pass over all four result states plus the
  red-flag interstitial, findings written down.
- A readability target on user-facing copy — Grade 8 is the usual
  health-literacy benchmark — checked in CI the way contrast already is.
- Fix the documented no-JavaScript gap in the small-screen menu. The
  `<details>` objection in `base.html` is real, but a nav that is a plain list
  below the breakpoint sidesteps it entirely.

**Done when** the "no screen reader has been listened to" limitation can be
deleted from the README truthfully.

---

## Phase 3 — earn the claim

Not engineering, and the actual gate between "an unusually well-built demo" and
something a clinic could pilot.

### 3.1 Clinician review — external lead time

The README names this as the gating item and both safety modules carry the same
admission in their docstrings. Give the reviewer a defined scope so it's a
half-day, not a research project:

- tier-1 physiological limits and tier-2 red flags in `app/ml/safety.py`
- the emergency signal list in `app/ml/triage.py`
- the rubric weights in `app/ml/lifestyle.py`
- the result copy on all four pages

Record the reviewer, their registration, the date, and the commit reviewed, in
the repo. A review nobody can point to is worth what an unreviewed threshold is
worth.

**Done when** three docstrings and one README bullet stop saying "has NOT been
reviewed by a clinician."

### 3.2 Intended use — 1 day

One page, and the difference between a project and a product: who it's for; who
it is *not* for (under 18, pregnancy, an existing diagnosis of the condition
being screened); what decision it supports; and why it sits outside medical
device regulation — the FDA's clinical decision support criteria and EU MDR Rule
11 are the two frames that matter, and the app's design already satisfies the
important one, which is that a clinician can see and check the basis of every
recommendation.

The "not a medical device" line in the README is currently an assertion. This
turns it into an argument.

**Done when** `INTENDED_USE.md` exists, is linked from the disclaimer on every
page, and names the populations the app declines to assess.

### 3.3 Published model cards — 3 days · *much cheaper after #6, #7, #8*

`metadata.json` is now most of a model card: source, population, estimator,
weighted and unweighted metrics with baselines, subgroups six ways, risk bands
with intervals, decision threshold and its rule, provenance, library versions,
training date. It's sitting in a file nobody outside the repo will ever open.

Render it. A page at `/models/<name>` per model *and rule* — the NHANES table and
the lifestyle rubric have more to show than either model does. Link it from the
result page that uses it.

Almost nobody in this category publishes one, and for a project whose whole
argument is "it shows its working", it's the page that makes the argument.

**Done when** every result page links to the card for the thing that produced
it, and the card is generated from metadata rather than written by hand.

---

## Carried forward from the last session

Small, specific, and none of them urgent.

| | Item | Effort |
|---|---|---|
| a | **Close the CSP's `style-src`.** It still allows `'unsafe-inline'` because a nonce can't cover style *attributes* — there are **32** of them plus **5** per-page `<style>` blocks. Move them into stylesheets and the policy tightens to `style-src 'self'`. | 1 day |
| b | **Seven residual emergency false alarms**, all safe-direction and named in `tests/test_triage.py`. Six are `incidental` (an emergency phrase's words scattered through an unrelated sentence); one is `negated` ("no history of stroke or heart attack", where the cue sits past the window). Worth revisiting only with a bigger labelled set — the current 110 phrasings are enough to catch regressions, not enough to tune against. | — |
| c | **Trim the icon sprite.** 27 kB inlined into every page, against ~450 kB of third-party assets removed. Only worth it if page weight starts mattering; per-page subsetting is the obvious approach and adds build complexity. | 0.5 day |
| d | **`CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`** — all missing. `SECURITY.md` matters most: this is a health app and there is currently no stated way to report a vulnerability. | 0.5 day |
| e | **BRFSS annual refresh.** `fetch_brfss.py` takes `--year`; the 2024 cycle is the next one. The mapping is verified on fetch, so this should be a rerun rather than a project. | 0.5 day |

---

## What not to build

`ADVANCEMENT_ROADMAP.md` — gitignored, and its own header calls it stale —
proposes deep learning (TabNet, TabTransformer, LSTMs), ensembling the models
into a unified index, and splitting the app into microservices. Each would make
this product worse.

**TabNet on fifteen binary survey answers** will not beat a calibrated
HistGradientBoosting, and it would cost the calibration and the counterfactual
explanations that are this project's actual differentiators. **Ensembling** a
heart model fit on US survey respondents with a migraine model of unknown
provenance produces a number with no referent — you cannot state what a "unified
health index" is an estimate *of*. **Microservices** add operational surface to
an app that fits in one process, serves a prediction in milliseconds, and has no
state to coordinate.

The instinct behind all three is that more sophistication reads as more serious.
In health screening the opposite is true: the sleep model was deleted because a
lookup table matched it, and that decision is still the strongest thing in the
repository.

---

## The through-line

Every item above is the same move this project has now made six times: find the
place where a number is doing more work than its evidence supports, and either
strengthen the evidence or weaken the claim. The lifestyle model was deleted for
it. The sleep model was deleted for it. The heart model's headline was replaced
with ROC-AUC and a baseline, then re-scored against the country rather than the
survey, then split six ways to show who it works less well for. The emergency
check was measured instead of sampled.

Phase 2 and Phase 3 are the same instinct applied to the things that aren't code
— what the product remembers, and who has checked it.
