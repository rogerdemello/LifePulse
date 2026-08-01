"""Download and assemble the NHANES sleep dataset.

    python ml_model/fetch_nhanes.py

Writes ``data/nhanes_sleep.csv``.

Why this exists: the sleep model was trained on a small Kaggle file whose
respondents all had a systolic blood pressure between 110 and 144 and a resting
pulse between 60 and 89. Anyone hypertensive or bradycardic fell outside
everything the model had ever seen, so the app correctly refused to trust its
own answer for exactly the people most likely to have sleep apnea.

NHANES is the US National Health and Nutrition Examination Survey: a real,
public-domain, nationally representative survey with measured (not self-
reported) blood pressure, pulse and BMI. The 2017-18 cycle gives 5,500 adults
spanning systolic 72-228 and pulse 34-136.

**The labels are self-reported symptoms, not diagnoses.** NHANES has no
polysomnography. A respondent counts as showing apnea-type symptoms if they
report snorting, gasping or stopping breathing during sleep at least
occasionally, and insomnia-type symptoms if they have told a doctor they have
trouble sleeping. That is a screening signal of the same kind the STOP-BANG
questionnaire uses, and the app says so rather than claiming to detect apnea.

Source: https://wwwn.cdc.gov/nchs/nhanes/  (public domain, no licence needed)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "nhanes"
OUT = ROOT / "data" / "nhanes_sleep.csv"

# The 2017-2018 cycle ("_J"). This path serves the real transport files; the
# shorter /Nchs/Nhanes/2017-2018/ form returns an HTML page instead.
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

FILES = {
    "DEMO_J.xpt": "demographics",
    "SLQ_J.xpt": "sleep questionnaire",
    "BPX_J.xpt": "blood pressure & pulse (measured)",
    "BMX_J.xpt": "body measures",
}

# NHANES codes refusals and don't-knows as out-of-range sentinels. Left in
# place they would be read as real values -- 7 nights of snoring a week on a
# 0-3 scale, or a 99-year-old.
MISSING_CODES = (7, 9, 77, 99, 777, 999)

log = logging.getLogger("nhanes")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, description in FILES.items():
        target = RAW / filename
        if target.exists():
            log.info("  %-12s cached (%s)", filename, description)
            continue
        log.info("  %-12s downloading (%s)", filename, description)
        response = requests.get(BASE + filename, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"HEADER"):
            raise RuntimeError(
                f"{filename} is not a SAS transport file -- CDC may have moved it"
            )
        target.write_bytes(response.content)


def _read(filename):
    return pd.read_sas(RAW / filename, format="xport")


def _clean(series, missing=MISSING_CODES):
    """Blank NHANES sentinel codes, and round the float noise out of zeros."""
    cleaned = series.where(~series.isin(missing))
    return cleaned.round(6)


def build():
    demo = _read("DEMO_J.xpt")[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    slq = _read("SLQ_J.xpt")[["SEQN", "SLD012", "SLQ030", "SLQ040", "SLQ050", "SLQ120"]]
    bpx = _read("BPX_J.xpt")[["SEQN", "BPXSY1", "BPXDI1", "BPXPLS"]]
    bmx = _read("BMX_J.xpt")[["SEQN", "BMXBMI"]]

    df = demo.merge(slq, on="SEQN").merge(bpx, on="SEQN").merge(bmx, on="SEQN")
    df = df[df.RIDAGEYR >= 18]

    for column in ["SLQ030", "SLQ040", "SLQ050", "SLQ120"]:
        df[column] = _clean(df[column])
    # A diastolic reading of 0 is a failed measurement, not a live patient.
    df["BPXDI1"] = df["BPXDI1"].where(df["BPXDI1"] > 20)

    tidy = pd.DataFrame({
        "Age": df.RIDAGEYR,
        # NHANES: 1 = male, 2 = female.
        "Gender": df.RIAGENDR.map({1: "Male", 2: "Female"}),
        "Sleep Hours": df.SLD012,
        "Snoring": df.SLQ030,            # 0 never .. 3 frequently
        "Daytime Sleepiness": df.SLQ120,  # 0 never .. 4 almost always
        "BMI": df.BMXBMI,
        "Systolic": df.BPXSY1,
        "Diastolic": df.BPXDI1,
        "Pulse": df.BPXPLS,
        "_gasping": df.SLQ040,            # label source, never a feature
        "_told_doctor": df.SLQ050,
    })

    # Apnea takes precedence over insomnia where both are reported: it is the
    # more urgent of the two to raise with a doctor.
    # "No symptoms" rather than "None": pandas parses the literal string "None"
    # as NaN when reading the CSV back, which silently deletes the majority
    # class between writing this file and training on it.
    apnea = tidy._gasping >= 2
    insomnia = tidy._told_doctor == 1
    tidy["Sleep Problem"] = "No symptoms"
    tidy.loc[insomnia, "Sleep Problem"] = "Insomnia symptoms"
    tidy.loc[apnea, "Sleep Problem"] = "Apnea symptoms"

    tidy = tidy.drop(columns=["_gasping", "_told_doctor"])
    tidy = tidy.dropna()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT, index=False)
    return tidy


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("NHANES 2017-2018")
    download()
    tidy = build()
    log.info("")
    log.info("wrote %s: %d adults x %d columns", OUT.name, len(tidy), tidy.shape[1])
    log.info("  classes: %s", tidy["Sleep Problem"].value_counts().to_dict())
    for column in ["Systolic", "Diastolic", "Pulse", "BMI", "Age"]:
        series = tidy[column]
        log.info("  %-10s %6.1f .. %6.1f", column, series.min(), series.max())
    return tidy


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
