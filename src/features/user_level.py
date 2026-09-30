"""Per-user aggregation of transaction-level data into one row per user."""
import numpy as np
import pandas as pd


def _entropy(counts: pd.Series) -> float:
    """Shannon entropy (natural log) of a distribution of counts."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _mode(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def _burstiness(gaps: np.ndarray) -> float:
    """Classic burstiness parameter B = (std - mean) / (std + mean); 0 if undefined."""
    if len(gaps) < 2:
        return 0.0
    mean = gaps.mean()
    std = gaps.std()
    if mean + std == 0:
        return 0.0
    return float((std - mean) / (std + mean))


def _activity_features(g: pd.DataFrame, dataset_max_date: pd.Timestamp) -> dict:
    dates = g["transaction_date"].sort_values()
    tenure_days = (dates.max() - dates.min()).days
    gaps = dates.diff().dropna().dt.days.to_numpy()
    return {
        "n_transactions": len(g),
        "active_days": dates.nunique(),
        "active_months": g["year_month"].nunique(),
        "tenure_days": int(tenure_days),
        "recency_days": int((dataset_max_date - dates.max()).days),
        "avg_gap_days": float(gaps.mean()) if len(gaps) else 0.0,
        "burstiness": _burstiness(gaps),
    }


def _value_features(valid: pd.DataFrame) -> dict:
    if valid.empty:
        return {"total_net_amount": 0.0, "avg_net_amount": 0.0, "median_net_amount": 0.0,
                "max_net_amount": 0.0, "cv_net_amount": 0.0}
    amt = valid["net_amount"]
    mean = amt.mean()
    std = amt.std()
    return {
        "total_net_amount": float(amt.sum()),
        "avg_net_amount": float(mean),
        "median_net_amount": float(amt.median()),
        "max_net_amount": float(amt.max()),
        "cv_net_amount": float(std / mean) if mean else 0.0,
    }


def _category_features(valid: pd.DataFrame, mcc_groups: list[str], mcc_social: list[str], mcc_essential: list[str]) -> dict:
    n_valid = len(valid)
    out = {
        "n_categories": valid["merchant_category_id"].nunique(),
        "category_entropy": _entropy(valid["merchant_category_id"].value_counts()) if n_valid else 0.0,
        "n_merchants": valid["merchant_id"].nunique(),
    }
    n_merchants = out["n_merchants"]
    out["merchant_diversity"] = _entropy(valid["merchant_id"].value_counts()) if n_valid else 0.0
    out["merchant_repeat_ratio"] = float((n_valid - n_merchants) / n_valid) if n_valid else 0.0

    group_counts = valid["mcc_group"].value_counts()
    for group in mcc_groups:
        out[f"share_{group}"] = float(group_counts.get(group, 0) / n_valid) if n_valid else 0.0

    out["social_share"] = float(valid["merchant_category_id"].isin(mcc_social).mean()) if n_valid else 0.0
    out["essential_share"] = float(valid["merchant_category_id"].isin(mcc_essential).mean()) if n_valid else 0.0
    return out


def _geo_features(g: pd.DataFrame) -> dict:
    city_counts = g["city"].value_counts()
    primary_city = city_counts.index[0] if len(city_counts) else None
    primary_city_share = float(city_counts.iloc[0] / len(g)) if len(city_counts) else 0.0

    province_counts = g["province"].value_counts()
    primary_province = province_counts.index[0] if len(province_counts) else None

    region_counts = g["region_group"].value_counts()
    primary_region_group = region_counts.index[0] if len(region_counts) else None

    n_cities = g["city"].nunique()
    n_provinces = g["province"].nunique()
    return {
        "primary_city": primary_city,
        "primary_city_share": primary_city_share,
        "primary_province": primary_province,
        "primary_region_group": primary_region_group,
        "n_cities": n_cities,
        "n_provinces": n_provinces,
        "is_multi_city": n_cities > 1,
        "is_multi_province": n_provinces > 1,
    }


def _payment_promo_features(g: pd.DataFrame, valid: pd.DataFrame) -> dict:
    n = len(g)
    cc_share = float((g["payment_method"] == "credit_card").mean()) if n else 0.0
    if cc_share == 0:
        profile = "balance_only"
    elif cc_share == 1:
        profile = "credit_card_only"
    else:
        profile = "mixed"

    n_valid = len(valid)
    discount_share = float(valid["discount_applied"].fillna(False).mean()) if n_valid else 0.0
    total_promo = float(valid["promo_amount"].fillna(0).sum())
    used = valid[valid["discount_applied"].fillna(False)]
    avg_discount_rate = float(used["discount_rate"].mean()) if len(used) else 0.0

    return {
        "credit_card_share": cc_share,
        "payment_profile": profile,
        "discount_tx_share": discount_share,
        "total_promo_amount": total_promo,
        "avg_discount_rate_when_used": avg_discount_rate,
        "is_promo_user": bool(total_promo > 0),
        "loyalty_member": bool(g["loyalty_program"].fillna(False).any()),
    }


def _quality_features(g: pd.DataFrame) -> dict:
    n = len(g)
    n_failed = int(g["is_failed"].sum())
    n_completed = int(g["is_completed"].sum())
    n_refunded_of_completed = int((g["is_refunded"].fillna(False) & g["is_completed"]).sum())

    ratings = g["merchant_rating_clean"].dropna()
    return {
        "failure_rate": float(n_failed / n) if n else 0.0,
        "refund_rate": float(n_refunded_of_completed / n_completed) if n_completed else 0.0,
        "avg_rating": float(ratings.mean()) if len(ratings) else np.nan,
        "rating_std": float(ratings.std()) if len(ratings) > 1 else 0.0,
        "notes_share": float(g["transaction_notes"].fillna(False).mean()) if n else 0.0,
    }


def _time_features(g: pd.DataFrame, valid: pd.DataFrame, cfg: dict) -> dict:
    n_valid = len(valid)
    weekend_share = float(valid["is_weekend"].mean()) if n_valid else 0.0
    payday_share = float(valid["is_payday_window"].mean()) if n_valid else 0.0
    lebaran_share = float(valid["is_lebaran_period"].mean()) if n_valid else 0.0

    lebaran_start = pd.Timestamp(cfg["calendar"]["lebaran_start"])
    lebaran_end = pd.Timestamp(cfg["calendar"]["lebaran_end"])
    lebaran_days = (lebaran_end - lebaran_start).days + 1

    total_days = (valid["transaction_date"].max() - valid["transaction_date"].min()).days + 1 if n_valid else 0
    non_lebaran_days = max(total_days - lebaran_days, 1)

    n_lebaran = int(valid["is_lebaran_period"].sum())
    n_non_lebaran = n_valid - n_lebaran

    lebaran_rate = n_lebaran / lebaran_days
    non_lebaran_rate = n_non_lebaran / non_lebaran_days
    lebaran_lift = float(lebaran_rate / non_lebaran_rate) if non_lebaran_rate > 0 else (float("inf") if lebaran_rate > 0 else 0.0)

    return {
        "weekend_share": weekend_share,
        "payday_share": payday_share,
        "lebaran_share": lebaran_share,
        "lebaran_lift": lebaran_lift,
    }


def _device_features(g: pd.DataFrame) -> dict:
    return {
        "os_family": _mode(g["os_family"]),
        "device_type": _mode(g["device_type"]),
        "os_version": _mode(g["os_version"]),
        "is_legacy_os": bool(g["info_legacy_os"].fillna(False).any()),
    }


def build_user_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mcc_groups = sorted({v["group"] for v in cfg["mcc"].values()})
    mcc_social = cfg.get("mcc_social", [])
    mcc_essential = cfg.get("mcc_essential", [])
    dataset_max_date = df["transaction_date"].max()

    rows = []
    for user_id, g in df.groupby("user_id", sort=False):
        valid = g[g["is_valid_purchase"]]
        n_valid = len(valid)

        row = {"user_id": user_id}
        row.update(_activity_features(g, dataset_max_date))
        row["n_valid_purchases"] = n_valid
        row.update(_value_features(valid))
        row.update(_category_features(valid, mcc_groups, mcc_social, mcc_essential))
        row.update(_geo_features(g))
        row.update(_payment_promo_features(g, valid))
        row.update(_quality_features(g))
        row.update(_time_features(g, valid, cfg))
        row.update(_device_features(g))
        rows.append(row)

    return pd.DataFrame(rows)
