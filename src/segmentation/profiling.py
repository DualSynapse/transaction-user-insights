"""Turn raw cluster ids into named personas and per-segment summary statistics."""
import numpy as np
import pandas as pd

LABEL_PRIORITY = ["champion", "at_risk", "bargain_hunter", "new_low_engagement", "regular"]

LABEL_NAMES = {
    "champion": "Champions (High-Value Loyal)",
    "at_risk": "At-Risk / Dormant",
    "bargain_hunter": "Bargain Hunters",
    "new_low_engagement": "New / Low Engagement",
    "regular": "Regular / Steady Spenders",
}


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std else pd.Series(0.0, index=s.index)


def label_segments(cluster_means: pd.DataFrame) -> dict:
    """Assign a human-readable persona name to each cluster id from its centroid means.

    cluster_means: index = cluster id, columns must include recency_days, active_months,
    n_valid_purchases, avg_net_amount, discount_tx_share.
    """
    value_score = (
        _zscore(cluster_means["n_valid_purchases"])
        + _zscore(cluster_means["avg_net_amount"])
        + _zscore(cluster_means["active_months"])
        - _zscore(cluster_means["recency_days"])
    )

    assigned = {}
    remaining = list(cluster_means.index)

    def pop_extreme(candidates, series, take_max):
        if not candidates:
            return None
        sub = series.loc[candidates]
        return sub.idxmax() if take_max else sub.idxmin()

    champion = pop_extreme(remaining, value_score, take_max=True)
    assigned[champion] = "champion"
    remaining.remove(champion)

    at_risk = pop_extreme(remaining, value_score, take_max=False)
    assigned[at_risk] = "at_risk"
    remaining.remove(at_risk)

    if remaining:
        bargain = pop_extreme(remaining, cluster_means["discount_tx_share"], take_max=True)
        assigned[bargain] = "bargain_hunter"
        remaining.remove(bargain)

    if remaining:
        newest = pop_extreme(remaining, cluster_means["active_months"], take_max=False)
        assigned[newest] = "new_low_engagement"
        remaining.remove(newest)

    for cluster_id in remaining:
        assigned[cluster_id] = "regular"

    return {cid: LABEL_NAMES[key] for cid, key in assigned.items()}


def build_segment_profiles(user_features: pd.DataFrame, feature_cols: list[str], profiling_cols: list[str]) -> dict:
    """Per-segment size, share, numeric feature means, and top categorical profiling values."""
    n_total = len(user_features)
    profiles = {}

    numeric_profiling = [c for c in profiling_cols if pd.api.types.is_numeric_dtype(user_features[c])]
    categorical_profiling = [c for c in profiling_cols if c not in numeric_profiling]

    for seg_id, g in user_features.groupby("segment_id"):
        profile = {
            "segment_id": int(seg_id),
            "label": g["segment_label"].iloc[0],
            "n_users": len(g),
            "share_of_users": round(len(g) / n_total, 4),
            "feature_means": {c: round(float(g[c].mean()), 4) for c in feature_cols},
            "profiling_means": {c: round(float(g[c].mean()), 4) for c in numeric_profiling},
            "top_categories": {
                c: g[c].value_counts(normalize=True).head(3).round(4).to_dict()
                for c in categorical_profiling
            },
        }
        profiles[str(seg_id)] = profile

    return profiles
