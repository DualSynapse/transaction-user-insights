"""run_features(cfg): transaction-level features -> user-level features -> personality -> demographics."""
import logging

import pandas as pd

from src.features.demographics import add_demographic_signals, build_feasibility_table
from src.features.personality import add_personality_traits
from src.features.transaction_level import add_transaction_features
from src.features.user_level import build_user_features
from src.utils.tools import read_table, write_json, write_table

logger = logging.getLogger("pipeline")

FEATURE_DESCRIPTIONS = {
    "user_id": ("identity", "Unique user identifier"),
    "n_transactions": ("activity", "Total number of transactions (any status)"),
    "n_valid_purchases": ("activity", "Completed, non-refunded transactions"),
    "active_days": ("activity", "Distinct calendar days with a transaction"),
    "active_months": ("activity", "Distinct calendar months with a transaction"),
    "tenure_days": ("activity", "Days between the user's first and last transaction"),
    "recency_days": ("activity", "Days between the user's last transaction and the dataset's last date"),
    "avg_gap_days": ("activity", "Average gap in days between consecutive transactions"),
    "burstiness": ("activity", "Burstiness parameter of inter-transaction gaps (-1 regular, +1 bursty)"),
    "total_net_amount": ("value", "Sum of net (post-discount) amount over valid purchases"),
    "avg_net_amount": ("value", "Mean net amount over valid purchases"),
    "median_net_amount": ("value", "Median net amount over valid purchases"),
    "max_net_amount": ("value", "Maximum net amount over valid purchases"),
    "cv_net_amount": ("value", "Coefficient of variation of net amount over valid purchases"),
    "n_categories": ("category", "Distinct merchant categories (MCC) used"),
    "category_entropy": ("category", "Shannon entropy of the user's MCC distribution"),
    "n_merchants": ("category", "Distinct merchants used"),
    "merchant_diversity": ("category", "Shannon entropy of the user's merchant distribution"),
    "merchant_repeat_ratio": ("category", "Share of valid purchases that are repeats of a previously visited merchant"),
    "social_share": ("category", "Share of valid purchases at social/dining/entertainment merchants"),
    "essential_share": ("category", "Share of valid purchases at essential (grocery/pharmacy) merchants"),
    "primary_city": ("geography", "Most frequent transaction city for this user"),
    "primary_city_share": ("geography", "Share of transactions in the primary city"),
    "primary_province": ("geography", "Most frequent transaction province"),
    "primary_region_group": ("geography", "Most frequent region group"),
    "n_cities": ("geography", "Distinct cities transacted in"),
    "n_provinces": ("geography", "Distinct provinces transacted in"),
    "is_multi_city": ("geography", "Whether the user transacted in more than one city"),
    "is_multi_province": ("geography", "Whether the user transacted in more than one province"),
    "credit_card_share": ("payment", "Share of transactions paid by credit card"),
    "payment_profile": ("payment", "balance_only / credit_card_only / mixed"),
    "discount_tx_share": ("payment", "Share of valid purchases using a discount"),
    "total_promo_amount": ("payment", "Total promo amount received"),
    "avg_discount_rate_when_used": ("payment", "Mean discount rate on transactions where a discount was used"),
    "is_promo_user": ("payment", "Whether the user ever received a promo amount > 0"),
    "loyalty_member": ("payment", "Whether the user is flagged as a loyalty program member on any transaction"),
    "failure_rate": ("quality", "Share of transactions that failed"),
    "refund_rate": ("quality", "Share of completed transactions that were refunded"),
    "avg_rating": ("quality", "Mean rating given on completed transactions"),
    "rating_std": ("quality", "Standard deviation of ratings given"),
    "notes_share": ("quality", "Share of transactions with a note attached"),
    "weekend_share": ("time", "Share of valid purchases on a weekend"),
    "payday_share": ("time", "Share of valid purchases in the payday window"),
    "lebaran_share": ("time", "Share of valid purchases during the Eid al-Fitr period"),
    "lebaran_lift": ("time", "Ratio of the user's daily transaction rate during Eid al-Fitr vs. the rest of the period"),
    "os_family": ("device", "Most frequently used OS family"),
    "device_type": ("device", "Most frequently used device type"),
    "os_version": ("device", "Most frequently used OS version"),
    "is_legacy_os": ("device", "Whether the user was ever seen on a legacy OS version"),
    "home_city_estimate": ("demographics", "Estimated home city (= primary transaction city)"),
    "home_city_confidence": ("demographics", "Confidence in the home city estimate (none/low/medium/high)"),
    "income_proxy_tier": ("demographics", "Weak income proxy tier from credit-card usage and spend level"),
}

PERSONALITY_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def _build_feature_dictionary(user_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in user_features.columns:
        if col in FEATURE_DESCRIPTIONS:
            group, desc = FEATURE_DESCRIPTIONS[col]
        elif col.startswith("share_"):
            group, desc = "category", f"Share of valid purchases in MCC group '{col[len('share_'):]}'"
        elif any(col.startswith(f"{t}_") for t in PERSONALITY_TRAITS):
            trait = col.split("_")[0]
            group = "personality"
            desc = f"{trait.capitalize()} percentile score (0-100)" if col.endswith("_score") else f"{trait.capitalize()} raw z-score"
        elif col.endswith("_signal_count"):
            group, desc = "demographics", f"Number of valid purchases supporting {col[:-len('_count')]}"
        elif col.endswith("_signal_confidence"):
            group, desc = "demographics", f"Confidence level for {col[:-len('_confidence')]} (vs. chance baseline)"
        else:
            group, desc = "other", ""

        rows.append({
            "feature_name": col,
            "group": group,
            "description": desc,
            "dtype": str(user_features[col].dtype),
            "missing_share": round(float(user_features[col].isna().mean()), 4),
        })
    return pd.DataFrame(rows)


def run_features(cfg) -> pd.DataFrame:
    logger.info("Loading cleaned transactions")
    transactions = read_table(cfg.path("transactions_clean"))

    logger.info("Adding transaction-level features")
    transactions = add_transaction_features(transactions, cfg.raw)
    write_table(transactions, cfg.path("transactions_enriched"))

    logger.info("Building per-user features (%d users)", transactions["user_id"].nunique())
    user_features = build_user_features(transactions, cfg.raw)

    logger.info("Computing Big Five personality proxies")
    user_features, personality_meta = add_personality_traits(user_features)

    logger.info("Computing demographic signals")
    user_features = add_demographic_signals(user_features, transactions, cfg.raw)

    write_table(user_features, cfg.path("user_features"))
    logger.info("Wrote user features to %s (%d rows, %d cols)", cfg.path("user_features"), *user_features.shape)

    feature_dict = _build_feature_dictionary(user_features)
    feature_dict.to_csv(cfg.path("feature_dictionary"), index=False)

    feasibility = build_feasibility_table()
    feasibility.to_csv(cfg.path("demographic_feasibility"), index=False)

    meta = {
        "n_users": len(user_features),
        "n_features": user_features.shape[1] - 1,
        "personality": personality_meta,
    }
    write_json(meta, cfg.path("feature_meta"))
    logger.info("Wrote feature metadata to %s", cfg.path("feature_meta"))

    return user_features
