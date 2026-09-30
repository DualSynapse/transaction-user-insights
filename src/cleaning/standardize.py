"""Data types, category validation, booleans."""
import numpy as np
import pandas as pd

STRING_COLS = [
    "transaction_id", "user_id", "merchant_id", "merchant_name",
    "merchant_category_id", "geo_location", "payment_method", "user_agent",
    "loyalty_program", "discount_applied", "transaction_notes",
    "transaction_status", "is_refunded",
]

YES_NO_COLS = ["loyalty_program", "discount_applied", "transaction_notes", "is_refunded"]


def _tokens_to_nan(series: pd.Series, missing_tokens: list[str]) -> pd.Series:
    tokens_lower = {t.lower() for t in missing_tokens}
    stripped = series.astype(str).str.strip()
    is_hidden = stripped.str.lower().isin(tokens_lower)
    out = stripped.copy()
    out[is_hidden] = np.nan
    out[series.isna()] = np.nan
    return out


def standardize(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Trim whitespace, convert hidden-missing tokens to NaN, cast types, validate categories."""
    missing_tokens = cfg["cleaning"]["missing_tokens"]
    category_columns = cfg["cleaning"]["category_columns"]
    date_format = cfg["cleaning"]["date_format"]

    df = df_raw.copy()

    for col in STRING_COLS:
        if col in df.columns:
            df[col] = _tokens_to_nan(df[col], missing_tokens)

    # Numeric conversions
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df["promo_amount"] = pd.to_numeric(df["promo_amount"], errors="coerce")
    df["merchant_rating"] = pd.to_numeric(df["merchant_rating"], errors="coerce")

    # Date conversion (date only, no time)
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], format=date_format, errors="coerce"
    )
    df["dq_bad_date"] = df["transaction_date"].isna()

    # Category validation: flag values that are neither NaN nor in the allowed set
    for col, allowed in category_columns.items():
        if col not in df.columns:
            continue
        allowed_lower = {a.lower() for a in allowed}
        lower = df[col].str.lower()
        invalid = lower.notna() & ~lower.isin(allowed_lower)
        df[f"dq_invalid_{col}"] = invalid
        # normalize case for valid values
        df.loc[lower.isin(allowed_lower), col] = lower[lower.isin(allowed_lower)]

    # yes/no -> boolean (NaN stays NaN -> pandas nullable boolean)
    for col in YES_NO_COLS:
        mapped = df[col].map({"yes": True, "no": False})
        df[col] = mapped.astype("boolean")

    # transaction_status: three real-world states -> completed / failed / pending (status not yet resolved)
    status_lower = df["transaction_status"].str.lower()
    df["transaction_status_clean"] = status_lower.fillna("pending")
    df["is_completed"] = df["transaction_status_clean"] == "completed"
    df["is_pending"] = df["transaction_status_clean"] == "pending"
    df["is_failed"] = df["transaction_status_clean"] == "failed"

    return df
