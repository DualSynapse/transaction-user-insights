import numpy as np
import pandas as pd

from src.features.demographics import (
    _age_bracket_estimate,
    _education_signal,
    _gender_lean_signal,
    _marital_family_proxy,
    _work_location_estimate,
    _working_status_estimate,
)

DEMO_CFG = {
    "demographics": {
        "gender_lean": {
            "female_lean_mcc": ["7230", "5651"],
            "male_lean_mcc": ["5541", "5941"],
        },
        "working_status": {
            "min_valid_purchases": 5,
            "payday_share_threshold": 0.3,
            "weekend_share_max": 0.5,
        },
        "marital_family_proxy": {"min_primary_city_share": 0.5},
    }
}


def _base_user_features():
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "n_valid_purchases": [10, 10, 0],
        "is_multi_city": [False, True, True],
        "primary_city": ["A", "A", "B"],
        "primary_city_share": [1.0, 0.6, 0.3],
        "loyalty_member": [True, False, True],
        "payday_share": [0.5, 0.1, 0.0],
        "weekend_share": [0.2, 0.8, 0.0],
    })


def test_gender_lean_signal_direction():
    df = _base_user_features()
    valid = pd.DataFrame({
        "user_id": ["u1"] * 5 + ["u2"] * 5,
        "merchant_category_id": ["7230"] * 5 + ["5541"] * 5,  # u1 all female-lean, u2 all male-lean
    })
    out = _gender_lean_signal(df, valid, DEMO_CFG)
    assert out.loc[out["user_id"] == "u1", "gender_lean_signal"].iloc[0] == "female_lean"
    assert out.loc[out["user_id"] == "u2", "gender_lean_signal"].iloc[0] == "male_lean"
    assert out.loc[out["user_id"] == "u3", "gender_lean_confidence"].iloc[0] == "none"


def test_age_bracket_prioritizes_parental_over_student():
    df = _base_user_features()
    df["student_signal_count"] = [2, 0, 0]
    df["parental_signal_count"] = [0, 3, 0]
    df["student_signal_confidence"] = ["high", "none", "none"]
    df["parental_signal_confidence"] = ["none", "high", "none"]

    out = _age_bracket_estimate(df)
    assert out.loc[0, "age_bracket_estimate"] == "18-24 (student-leaning)"
    assert out.loc[1, "age_bracket_estimate"] == "25-45 (parent-leaning)"
    assert out.loc[2, "age_bracket_estimate"] == "unknown_adult"
    assert out.loc[2, "age_bracket_confidence"] == "none"


def test_education_signal_from_student_mcc():
    df = _base_user_features()
    df["student_signal_count"] = [1, 0, 0]
    df["student_signal_confidence"] = ["medium", "none", "none"]

    out = _education_signal(df)
    assert out.loc[0, "education_signal"] == "likely_in_tertiary_education"
    assert out.loc[0, "education_confidence"] == "medium"
    assert out.loc[1, "education_signal"] == "no_signal"


def test_work_location_differs_from_home_when_weekday_pattern_differs():
    df = _base_user_features()
    # u2 spends weekdays in city "C" but overall primary city is "A" (per _base_user_features)
    valid = pd.DataFrame({
        "user_id": ["u2"] * 10,
        "city": ["C"] * 8 + ["A"] * 2,
        "is_weekend": [False] * 8 + [True] * 2,
    })
    out = _work_location_estimate(df, valid)
    u2 = out[out["user_id"] == "u2"].iloc[0]
    assert u2["work_location_estimate"] == "C"
    assert u2["work_location_confidence"] in {"medium", "high"}

    # u1 is not multi-city -> work location falls back to home city with low confidence
    u1 = out[out["user_id"] == "u1"].iloc[0]
    assert u1["work_location_estimate"] == "A"
    assert u1["work_location_confidence"] == "low"


def test_working_status_requires_enough_data_and_payday_pattern():
    df = _base_user_features()
    out = _working_status_estimate(df, DEMO_CFG)
    assert out.loc[0, "working_status_estimate"] == "likely_employed_regular_income"
    assert out.loc[1, "working_status_estimate"] == "irregular_or_unknown"  # weekend_share too high
    assert out.loc[2, "working_status_estimate"] == "insufficient_signal"  # n_valid_purchases = 0


def test_marital_family_proxy_requires_all_three_signals():
    df = _base_user_features()
    df["parental_signal_count"] = [1, 1, 1]
    out = _marital_family_proxy(df, DEMO_CFG)
    # u1: parental>0, loyalty=True, primary_city_share=1.0 >= 0.5 -> family proxy
    assert out.loc[0, "marital_family_proxy"] == "family_proxy_likely"
    # u2: loyalty=False -> no signal despite parental + settled enough
    assert out.loc[1, "marital_family_proxy"] == "no_signal"
    # u3: primary_city_share=0.3 < 0.5 -> not settled enough
    assert out.loc[2, "marital_family_proxy"] == "no_signal"
