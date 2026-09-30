"""MCC group, net amount, time flags — transaction-level derived columns."""
import pandas as pd


def add_transaction_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    mcc_cfg = cfg["mcc"]

    df["mcc_description"] = df["merchant_category_id"].map(
        lambda m: mcc_cfg.get(m, {}).get("description", "Unknown")
    )
    df["mcc_group"] = df["merchant_category_id"].map(
        lambda m: mcc_cfg.get(m, {}).get("group", "Other")
    )
    df["info_mcc_non_standard"] = df["merchant_category_id"] == "5412"

    df["net_amount"] = df["transaction_amount"] - df["promo_amount"].fillna(0)
    df["discount_rate"] = (df["promo_amount"] / df["transaction_amount"]).where(
        df["transaction_amount"] > 0
    )

    df["is_valid_purchase"] = df["is_completed"] & ~(df["is_refunded"].fillna(False))
    df["is_discount_clean"] = df["discount_applied"] & ~df["dq_discount_zero_promo"].fillna(False)

    df["year_month"] = df["transaction_date"].dt.to_period("M").astype(str)
    df["week"] = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["is_weekend"] = df["transaction_date"].dt.dayofweek >= 5

    payday_days = set(cfg["calendar"]["payday_days"])
    df["is_payday_window"] = df["transaction_date"].dt.day.isin(payday_days)

    lebaran_start = pd.Timestamp(cfg["calendar"]["lebaran_start"])
    lebaran_end = pd.Timestamp(cfg["calendar"]["lebaran_end"])
    df["is_lebaran_period"] = df["transaction_date"].between(lebaran_start, lebaran_end)

    return df
