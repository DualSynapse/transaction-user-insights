"""Demographic signals + feasibility table.

Columns are created only for attributes with a supporting signal in the transaction data
(a specific MCC that plausibly indicates the attribute). Everything else is documented as
infeasible in demographic_feasibility.csv instead of being fabricated.
"""
import numpy as np
import pandas as pd

SIGNALS = {
    "parental_signal": ["8211"],
    "student_signal": ["8220"],
    "vehicle_signal": ["5541"],
    "homeowner_signal": ["5251", "5712"],
    "health_care_signal": ["5912"],
    "active_lifestyle_signal": ["5941"],
}

INFEASIBLE_ATTRIBUTES = [
    {"attribute": "gender", "reason": "No self-reported field and no MCC or behavior pattern in this dataset reliably predicts gender.",
     "data_needed": "Self-reported profile field, or a validated third-party demographic append."},
    {"attribute": "age", "reason": "No birthdate/age field; spending category alone is a very weak and biased age proxy.",
     "data_needed": "Self-reported profile field (birthdate) or KYC data."},
    {"attribute": "education", "reason": "MCC 8220 (universities) indicates possible current study, not attained education level.",
     "data_needed": "Self-reported education field."},
    {"attribute": "marital_status", "reason": "No transaction pattern in this dataset is a validated proxy for marital status.",
     "data_needed": "Self-reported profile field."},
    {"attribute": "work_location", "reason": "geo_location is the transaction point, not a home/work address; no separate work-hours location signal is available.",
     "data_needed": "Registered address field, or GPS pings with timestamps dense enough to infer a work cluster."},
    {"attribute": "working_status", "reason": "No income, payroll, or employer-linked transaction data is present.",
     "data_needed": "Payroll deposit data or self-reported employment field."},
    {"attribute": "industry", "reason": "No employer or occupation field exists in the dataset.",
     "data_needed": "Self-reported occupation/industry field, or employer-linked payroll data."},
]


def _confidence(count: pd.Series, expected: pd.Series) -> pd.Series:
    ratio = count / expected.replace(0, np.nan)
    conf = pd.Series("none", index=count.index)
    conf[(count >= 1) & (ratio <= 1.5)] = "low"
    conf[(count >= 1) & (ratio > 1.5) & (ratio <= 3)] = "medium"
    conf[(count >= 1) & (ratio > 3)] = "high"
    conf[count == 0] = "none"
    conf[expected.isna() | (expected == 0)] = "low"
    conf[count == 0] = "none"
    return conf


def add_demographic_signals(user_features: pd.DataFrame, transactions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = user_features.copy()
    valid = transactions[transactions["is_valid_purchase"]]
    n_valid_total = len(valid)

    mcc_counts_global = valid["merchant_category_id"].value_counts()

    for signal_name, mcc_list in SIGNALS.items():
        global_p = mcc_counts_global.reindex(mcc_list, fill_value=0).sum() / n_valid_total if n_valid_total else 0.0

        counts = (
            valid[valid["merchant_category_id"].isin(mcc_list)]
            .groupby("user_id")
            .size()
            .reindex(df["user_id"], fill_value=0)
            .to_numpy()
        )
        df[f"{signal_name}_count"] = counts

        expected = df["n_valid_purchases"] * global_p
        df[f"{signal_name}_confidence"] = _confidence(
            pd.Series(counts, index=df.index), expected
        ).to_numpy()

    def city_confidence(share):
        if share >= 0.8:
            return "high"
        if share >= 0.5:
            return "medium"
        return "low"

    df["home_city_estimate"] = df["primary_city"]
    df["home_city_confidence"] = df["primary_city_share"].apply(city_confidence)

    amount_pct = df["avg_net_amount"].rank(pct=True)

    def income_tier(cc_share, pct):
        if cc_share > 0.5 and pct >= 0.66:
            return "high"
        if cc_share == 0 and pct <= 0.33:
            return "low"
        return "medium"

    df["income_proxy_tier"] = [
        income_tier(cc, pct) for cc, pct in zip(df["credit_card_share"], amount_pct)
    ]

    return df


def build_feasibility_table() -> pd.DataFrame:
    return pd.DataFrame(INFEASIBLE_ATTRIBUTES)
