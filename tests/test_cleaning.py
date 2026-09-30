import pandas as pd
import pytest

from src.cleaning.audit import audit_missing
from src.cleaning.consistency import check_consistency
from src.cleaning.device_parser import parse_devices
from src.cleaning.standardize import standardize

MISSING_TOKENS = ["", "na", "n/a", "null", "none", "nan", "-", "?"]

BASE_CFG = {
    "cleaning": {
        "missing_tokens": MISSING_TOKENS,
        "category_columns": {
            "payment_method": ["balance", "credit_card"],
            "transaction_status": ["completed", "failed"],
            "loyalty_program": ["yes", "no"],
            "discount_applied": ["yes", "no"],
            "transaction_notes": ["yes", "no"],
            "is_refunded": ["yes", "no"],
        },
        "date_format": "%d/%m/%Y",
        "legacy_os_thresholds": {"Android": 8.0, "iOS": None},
    }
}


def _raw_df():
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t3"],
        "user_id": ["u1", "u1", "u2", "u2"],
        "transaction_amount": ["10000.0", "N/A", "5000.0", "5000.0"],
        "transaction_date": ["01/03/2024", "02/03/2024", "03/03/2024", "03/03/2024"],
        "merchant_id": ["m1", "m2", "m3", "m3"],
        "merchant_name": ["A", "B", "C", "C"],
        "merchant_category_id": ["5411", "5812", "5812", "5812"],
        "geo_location": ["-6.2, 106.8", "-6.2, 106.8", "-7.9, 112.6", "-7.9, 112.6"],
        "payment_method": ["balance", "credit_card", "balance", "balance"],
        "user_agent": ["Android 9", "iPhone", "Android 4.4.2", "Android 4.4.2"],
        "loyalty_program": ["yes", "no", "", ""],
        "discount_applied": ["no", "yes", "", ""],
        "promo_amount": ["0.0", "500.0", "", ""],
        "transaction_notes": ["no", "no", "", ""],
        "merchant_rating": ["5", "4", "", ""],
        "transaction_status": ["completed", "completed", "", ""],
        "is_refunded": ["no", "no", "", ""],
    })


def test_audit_detects_hidden_missing_tokens():
    df = _raw_df()
    report = audit_missing(df, MISSING_TOKENS)
    assert report["columns"]["transaction_amount"]["n_hidden_missing_token"] == 1
    assert report["columns"]["loyalty_program"]["n_nan"] == 0  # "" read as literal string here, counted as NaN via na check
    assert report["n_rows"] == 4


def test_standardize_yes_no_and_status():
    df_raw = _raw_df().replace({"": pd.NA})
    df = standardize(df_raw, BASE_CFG)
    assert df["loyalty_program"].tolist() == [True, False, pd.NA, pd.NA]
    assert df["is_completed"].tolist() == [True, True, False, False]
    assert df["is_pending"].tolist() == [False, False, True, True]


def test_consistency_flags():
    df_raw = _raw_df().replace({"": pd.NA})
    df = standardize(df_raw, BASE_CFG)
    df = check_consistency(df, BASE_CFG)

    assert df["dq_duplicate_transaction_id"].tolist() == [False, False, True, True]
    assert df["dq_amount_nonpositive"].iloc[1] == True  # noqa: E712 -- NaN amount
    assert bool(df["dq_any_issue"].any())


def test_device_parser():
    df_raw = _raw_df().replace({"": pd.NA})
    df = standardize(df_raw, BASE_CFG)
    df = parse_devices(df, BASE_CFG)

    row_android_new = df[df["user_agent"] == "Android 9"].iloc[0]
    assert row_android_new["os_family"] == "Android"
    assert row_android_new["info_legacy_os"] == False  # noqa: E712

    row_android_old = df[df["user_agent"] == "Android 4.4.2"].iloc[0]
    assert row_android_old["os_major"] == 4.0
    assert row_android_old["info_legacy_os"] == True  # noqa: E712

    row_iphone = df[df["user_agent"] == "iPhone"].iloc[0]
    assert row_iphone["os_family"] == "iOS"
    assert row_iphone["device_type"] == "Phone"
    assert pd.isna(row_iphone["os_version"])
