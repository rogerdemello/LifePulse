"""Prove the training path and the serving path produce identical features.

The original bug was not that either path was individually wrong -- each looked
reasonable in isolation. It was that they were two separate implementations
that drifted. These tests take one record through both entry points and assert
the resulting feature vectors are the same.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import features as F
from conftest import DATA, requires_dataset

# health_score is absent: it is scored by a rubric now, not a model, so there
# is no training path for a serving path to drift from.
CSV = {
    "heart": "heart_disease_health_indicators_BRFSS2015.csv",
    "sleep": "Sleep_health_and_lifestyle_dataset (1).csv",
    "migraine": "migraine_dataset_500 (1).csv",
}


def _load(name, rows=25):
    df = pd.read_csv(DATA / CSV[name], nrows=rows)
    df.columns = df.columns.str.strip()
    return df


@pytest.mark.parametrize("name", sorted(CSV))
def test_csv_row_and_form_dict_agree(name):
    """A DataFrame row and the equivalent dict must build the same vector.

    Training passes a whole DataFrame; a request passes one dict. If those two
    code paths could ever disagree, the serving features would drift from the
    trained ones -- which is precisely what happened before.
    """
    for csv_file in [CSV[name]]:
        if not (DATA / csv_file).exists():
            pytest.skip(f"{csv_file} is gitignored training data")

    df = _load(name)
    frame_features = F.BUILDERS[name](df)

    for i in range(len(df)):
        row = df.iloc[i]
        raw = {field: row[field] for field in _source_fields(name, df)}
        dict_features = F.BUILDERS[name](raw)
        np.testing.assert_allclose(
            dict_features.to_numpy()[0],
            frame_features.to_numpy()[i],
            rtol=1e-9,
            err_msg=f"{name}: row {i} differs between frame and dict paths",
        )


def _source_fields(name, df):
    fields = list(F.RAW_FIELDS[name])
    if name == "sleep":
        # build_sleep accepts either the CSV's "120/80" string or two numbers.
        fields += ["Blood Pressure"] if "Blood Pressure" in df.columns else []
    return fields


@requires_dataset(CSV["sleep"])
def test_blood_pressure_string_and_numbers_agree():
    """The form posts two numbers; the CSV carries "132/87". Same result."""
    base = {
        "Gender": "Male", "Age": 40, "Sleep Duration": 6.5,
        "Quality of Sleep": 7, "Physical Activity Level": 50,
        "Stress Level": 5, "Heart Rate": 72, "Daily Steps": 7000,
        "BMI Category": "Normal",
    }
    from_string = F.build_sleep({**base, "Blood Pressure": "132/87"})
    from_numbers = F.build_sleep({**base, "Systolic": 132, "Diastolic": 87})
    pd.testing.assert_frame_equal(from_string, from_numbers)


@requires_dataset(CSV["sleep"])
def test_normal_weight_and_normal_are_the_same_band():
    """The source records the same BMI band under two labels."""
    base = {
        "Gender": "Male", "Age": 40, "Sleep Duration": 6.5,
        "Quality of Sleep": 7, "Physical Activity Level": 50,
        "Stress Level": 5, "Heart Rate": 72, "Daily Steps": 7000,
        "Systolic": 120, "Diastolic": 80,
    }
    a = F.build_sleep({**base, "BMI Category": "Normal"})
    b = F.build_sleep({**base, "BMI Category": "Normal Weight"})
    pd.testing.assert_frame_equal(a, b)


@requires_dataset(CSV["migraine"])
def test_en_dash_activity_labels_are_understood():
    """The dataset writes '3–5 days/week' with an en-dash (U+2013).

    The previous training script mapped the ASCII '3-5 days/week', so 1,686 of
    2,000 rows became NaN and were filled with a constant -- the feature was
    dead in the shipped model, while the route fed it real values.
    """
    df = pd.read_csv(DATA / CSV["migraine"])
    df.columns = df.columns.str.strip()
    built = F.build_migraine(df)

    activity = built["Physical Activity"]
    assert activity.nunique() == 4, "Physical Activity collapsed to a constant"
    assert set(activity.unique()) == {0.0, 1.0, 2.0, 3.0}

    # Both dash forms and the numeric code must land on the same value.
    base = {
        "Age": 30, "Gender": "Female", "Sleep Hours": 7, "Water Intake": 5,
        "Skipped Meals": "No", "Caffeine": 2, "Stress": 5, "Screen Time": 6,
        "Menstruating": "No",
    }
    en_dash = F.build_migraine({**base, "Physical Activity": "3–5 days/week"})
    ascii_ = F.build_migraine({**base, "Physical Activity": "3-5 days/week"})
    numeric = F.build_migraine({**base, "Physical Activity": 2})
    pd.testing.assert_frame_equal(en_dash, ascii_)
    pd.testing.assert_frame_equal(en_dash, numeric)
