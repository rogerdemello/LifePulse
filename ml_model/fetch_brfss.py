"""Download and assemble a recent BRFSS cycle for the heart model.

    python ml_model/fetch_brfss.py            # 2023
    python ml_model/fetch_brfss.py --year 2022

Writes ``data/brfss_heart.csv``.

The heart model was trained on a pre-cleaned Kaggle derivative of BRFSS 2015 --
a decade old, and with no way to refresh it, because whoever built that file
did the variable mapping and never wrote it down. This script does the mapping
in the open against CDC's own release, so the model can move to a newer cycle
whenever one lands.

BRFSS is the Behavioral Risk Factor Surveillance System: a telephone survey the
CDC runs every year across all US states, around 430,000 respondents. Answers
are self-reported, which matters -- "have you ever been told you have high
blood pressure" is not a measurement.

**Fruits and Veggies are gone.** The 2015 file had them and the 2023 cycle does
not run that module at all. Before working around it, the question was whether
they earned their place: on the 2015 data, dropping both moves ROC-AUC from
0.8485 to 0.8486. They were two questions on the form buying nothing, so the
newer cycle costs nothing to adopt.

The download is 93 MB zipped and about 1.2 GB unpacked, so it is read in chunks
and only the columns below are kept.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "brfss"
OUT = ROOT / "data" / "brfss_heart.csv"

CHUNK_ROWS = 50_000

# BRFSS variable -> what it becomes. Codings are documented per column in
# `tidy()`; they are stable across recent cycles but not guaranteed, which is
# why the script verifies its output rather than trusting the mapping.
COLUMNS = [
    "_MICHD", "_RFHYPE6", "_RFCHOL3", "_CHOLCH3", "_BMI5", "SMOKE100",
    "CVDSTRK3", "DIABETE4", "_TOTINDA", "_RFDRHV8", "GENHLTH", "MENTHLTH",
    "PHYSHLTH", "DIFFWALK", "_SEX", "_AGEG5YR",
    # --- survey design, not features ---
    "_LLCPWT", "_STSTR",
]

# Why the design variables are here at all.
#
# BRFSS is not a simple random sample. States buy different sample sizes,
# landline and cell frames are drawn separately, and the whole thing is raked
# to census margins afterwards. A row is not one American; it is one American
# multiplied by _LLCPWT, which ranges from 0.16 to 69,786 in the 2023 cycle.
#
# Ignoring that is not a rounding error. On this file:
#
#     unweighted prevalence of _MICHD    0.0847
#     weighted prevalence                0.0647
#
# The app shows people a percentage and tells them it is literal, so it has to
# be a percentage *of something they belong to*. Trained unweighted, the model
# is calibrated to the population of people who answer landline surveys, which
# skews old -- and it was quoting that group's 9% rate back to every visitor as
# though it described US adults.
#
# _LLCPWT is the final combined landline-and-cell weight, which is the right
# one for the core variables used here. (_LLCPWT2 is for the split-questionnaire
# version-2 modules; none of ours come from those.)
#
# _PSU is deliberately NOT carried. It looks like a cluster identifier -- 26,444
# distinct values across 433,323 records -- but PSU ids are numbered within
# stratum, and every (_STSTR, _PSU) pair in the 2023 file is unique. There is
# exactly one record per cluster, so there is nothing to group a split by, and a
# "cluster-aware" split would be ceremony rather than rigour. _STSTR is kept
# because design-based standard errors need it, and the confidence interval on
# the displayed risk is the next thing this model owes its users.

log = logging.getLogger("brfss")


def download(year):
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"LLCP{year}.XPT"
    if target.exists():
        log.info("  cached: %s (%d MB)", target.name, target.stat().st_size // 1_000_000)
        return target

    url = f"https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
    log.info("  downloading %s", url)
    response = requests.get(url, timeout=900)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    # CDC ships the member name with a trailing space in some years.
    member = archive.namelist()[0]
    target.write_bytes(archive.read(member))
    log.info("  extracted %s (%d MB)", target.name, target.stat().st_size // 1_000_000)
    return target


def load(path):
    """Read only the columns we need, in chunks -- the file is over a gigabyte."""
    frames = []
    for chunk in pd.read_sas(path, format="xport", chunksize=CHUNK_ROWS):
        missing = [c for c in COLUMNS if c not in chunk.columns]
        if missing:
            raise SystemExit(
                f"BRFSS {path.stem} does not contain {missing}. Variable names "
                f"change between cycles -- check the codebook for that year."
            )
        frames.append(chunk[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _blank(series, codes):
    """Blank BRFSS 'refused' and 'don't know' sentinels."""
    return series.where(~series.isin(codes))


def tidy(raw):
    df = pd.DataFrame(index=raw.index)

    # _MICHD: 1 = reported MI or coronary heart disease, 2 = did not.
    df["HeartDiseaseorAttack"] = (raw["_MICHD"] == 1).astype(int)

    # Computed risk-factor variables: 1 = no, 2 = yes, 9 = don't know/refused.
    for source, name in [("_RFHYPE6", "HighBP"), ("_RFCHOL3", "HighChol"),
                         ("_RFDRHV8", "HvyAlcoholConsump")]:
        cleaned = _blank(raw[source], [9])
        df[name] = (cleaned == 2).astype("float64").where(cleaned.notna())

    # _CHOLCH3: 1 = checked within 5 years, 2/3 = not, 9 = missing.
    cholcheck = _blank(raw["_CHOLCH3"], [9])
    df["CholCheck"] = (cholcheck == 1).astype("float64").where(cholcheck.notna())

    # _BMI5 carries two implied decimals.
    df["BMI"] = raw["_BMI5"] / 100.0

    # SMOKE100: 1 = smoked 100+ cigarettes in life, 2 = no.
    smoke = _blank(raw["SMOKE100"], [7, 9])
    df["Smoker"] = (smoke == 1).astype("float64").where(smoke.notna())

    # CVDSTRK3 / DIFFWALK: 1 = yes, 2 = no.
    for source, name in [("CVDSTRK3", "Stroke"), ("DIFFWALK", "DiffWalk")]:
        cleaned = _blank(raw[source], [7, 9])
        df[name] = (cleaned == 1).astype("float64").where(cleaned.notna())

    # DIABETE4: 1 = yes, 2 = only during pregnancy, 3 = no, 4 = pre-diabetes.
    # Collapsed to the 0/1/2 scale the model already uses, where 1 covers the
    # borderline states.
    diabetes = _blank(raw["DIABETE4"], [7, 9])
    df["Diabetes"] = diabetes.map({1: 2, 2: 1, 4: 1, 3: 0})

    # _TOTINDA: 1 = any leisure-time physical activity in the past month.
    activity = _blank(raw["_TOTINDA"], [9])
    df["PhysActivity"] = (activity == 1).astype("float64").where(activity.notna())

    df["GenHlth"] = _blank(raw["GENHLTH"], [7, 9])

    # MENTHLTH / PHYSHLTH: days in the past 30, with 88 meaning none.
    for source, name in [("MENTHLTH", "MentHlth"), ("PHYSHLTH", "PhysHlth")]:
        days = _blank(raw[source], [77, 99])
        df[name] = days.replace(88, 0)

    # _SEX: 1 = male, matching the 2015 file's convention.
    df["Sex"] = (raw["_SEX"] == 1).astype(int)

    # _AGEG5YR: 1-13 five-year bands, 14 = don't know/refused.
    df["Age"] = _blank(raw["_AGEG5YR"], [14])

    # Survey design. Carried through under plain names so nothing downstream has
    # to know BRFSS's variable naming, and kept out of the feature contract in
    # app/ml/features.py -- these describe how a respondent was sampled, not
    # anything about their health.
    df["SurveyWeight"] = raw["_LLCPWT"]
    df["Stratum"] = raw["_STSTR"]

    return df.dropna()


def weighted_prevalence(df):
    """Share of US adults with the outcome, per the survey's own weights."""
    return float(
        (df.HeartDiseaseorAttack * df.SurveyWeight).sum() / df.SurveyWeight.sum()
    )


def verify(df):
    """Sanity-check the mapping rather than trusting it.

    A miscoded variable here would silently poison the model, and the failure
    would look like "the new cycle is just worse" rather than like a bug.
    """
    problems = []
    prevalence = df.HeartDiseaseorAttack.mean()
    if not 0.03 <= prevalence <= 0.20:
        problems.append(f"implausible outcome prevalence {prevalence:.3f}")

    # The weight is what makes the outcome rate a statement about US adults
    # rather than about survey respondents, so it gets checked like any other
    # mapping. A weight of zero or below is meaningless; a NaN would silently
    # drop a respondent out of every weighted statistic downstream.
    if not (df.SurveyWeight > 0).all():
        problems.append("SurveyWeight has non-positive values")
    if df.SurveyWeight.isna().any():
        problems.append("SurveyWeight has missing values")
    weighted = weighted_prevalence(df)
    if not 0.02 <= weighted <= 0.15:
        problems.append(f"implausible weighted prevalence {weighted:.3f}")
    # Weighting should pull the rate down, because BRFSS over-represents the
    # older respondents who have most of the heart disease. If it ever pushes it
    # up, the weight is probably mapped to the wrong column.
    if weighted > prevalence:
        problems.append(
            f"weighted prevalence {weighted:.3f} exceeds unweighted "
            f"{prevalence:.3f} -- is _LLCPWT mapped correctly?"
        )
    if not 10 <= df.BMI.median() <= 45:
        problems.append(f"implausible median BMI {df.BMI.median():.1f}")
    if not df.Age.between(1, 13).all():
        problems.append("Age outside the 1-13 band range")
    if not df.GenHlth.between(1, 5).all():
        problems.append("GenHlth outside 1-5")
    for column in ["MentHlth", "PhysHlth"]:
        if not df[column].between(0, 30).all():
            problems.append(f"{column} outside 0-30 days")
    for column in ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                   "PhysActivity", "HvyAlcoholConsump", "DiffWalk", "Sex"]:
        if not df[column].isin([0, 1]).all():
            problems.append(f"{column} is not binary")
    if not df.Diabetes.isin([0, 1, 2]).all():
        problems.append("Diabetes outside 0-2")

    if problems:
        raise SystemExit("BRFSS mapping looks wrong:\n  " + "\n  ".join(problems))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("BRFSS %d", args.year)

    path = download(args.year)
    log.info("  reading (this takes a minute)")
    raw = load(path)
    log.info("  %d respondents", len(raw))

    df = tidy(raw)
    verify(df)

    df.to_csv(OUT, index=False)
    log.info("")
    log.info("wrote %s: %d rows x %d columns", OUT.name, len(df), df.shape[1])
    # Both, always. The gap between them is the whole reason the weight is here,
    # and printing only one invites the next person to quote whichever they saw.
    log.info("  heart disease prevalence: %.4f unweighted, %.4f weighted",
             df.HeartDiseaseorAttack.mean(), weighted_prevalence(df))
    log.info("  survey weight %.2f-%.1f, sums to %.0f adults",
             df.SurveyWeight.min(), df.SurveyWeight.max(), df.SurveyWeight.sum())
    log.info("  BMI %.1f-%.1f, median %.1f", df.BMI.min(), df.BMI.max(), df.BMI.median())
    return df


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
