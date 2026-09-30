"""Duplicates, range checks, cross-column consistency checks -> dq_* flags."""
import numpy as np
import pandas as pd


def check_consistency(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # Duplicates
    df["dq_duplicate_transaction_id"] = df["transaction_id"].duplicated(keep=False)

    # Range checks
    df["dq_amount_nonpositive"] = df["transaction_amount"].isna() | (df["transaction_amount"] <= 0)
    df["dq_promo_negative"] = df["promo_amount"] < 0
    df["dq_promo_exceeds_amount"] = (
        df["promo_amount"].notna()
        & df["transaction_amount"].notna()
        & (df["promo_amount"] > df["transaction_amount"])
    )
    df["dq_rating_out_of_range"] = df["merchant_rating"].notna() & (
        (df["merchant_rating"] < 1) | (df["merchant_rating"] > 5)
    )

    # Cross-column consistency
    df["dq_rating_on_non_completed"] = df["merchant_rating"].notna() & ~df["is_completed"]
    df["dq_refund_on_non_completed"] = (df["is_refunded"] == True) & ~df["is_completed"]  # noqa: E712
    df["dq_discount_zero_promo"] = (df["discount_applied"] == True) & (  # noqa: E712
        df["promo_amount"].isna() | (df["promo_amount"] == 0)
    )

    # Clean rating: only meaningful for completed transactions (flag, don't delete original)
    df["merchant_rating_clean"] = df["merchant_rating"].where(df["is_completed"])

    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    df["dq_any_issue"] = df[dq_cols].any(axis=1)

    return df
