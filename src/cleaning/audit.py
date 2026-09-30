"""Missing-value and whitespace audit on the raw (string) data."""
import pandas as pd


def audit_missing(df_raw: pd.DataFrame, missing_tokens: list[str]) -> dict:
    """Count NaN, hidden missing tokens, and extra whitespace per column.

    df_raw must be read with dtype=str and no NA coercion (keep_default_na=False)
    so that hidden tokens and whitespace are visible.
    """
    tokens_lower = {t.lower() for t in missing_tokens}
    report = {}
    n_rows = len(df_raw)

    for col in df_raw.columns:
        series = df_raw[col]
        is_na = series.isna()
        stripped = series.fillna("").astype(str).str.strip()
        is_hidden_missing = stripped.str.lower().isin(tokens_lower) & ~is_na
        has_leading_trailing_ws = (
            series.fillna("").astype(str) != stripped
        ) & (stripped != "")

        report[col] = {
            "n_nan": int(is_na.sum()),
            "n_hidden_missing_token": int(is_hidden_missing.sum()),
            "n_whitespace_padded": int(has_leading_trailing_ws.sum()),
            "n_missing_total": int((is_na | is_hidden_missing).sum()),
            "pct_missing_total": round(float((is_na | is_hidden_missing).sum()) / n_rows * 100, 3) if n_rows else 0.0,
        }

    return {
        "n_rows": n_rows,
        "n_columns": len(df_raw.columns),
        "columns": report,
    }
