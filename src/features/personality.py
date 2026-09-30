"""Big Five proxies (mean z-score of signed indicators -> 0-100 percentile) + Cronbach's alpha."""
import numpy as np
import pandas as pd

TRAITS = {
    "openness": [("category_entropy", 1), ("n_categories", 1), ("merchant_diversity", 1)],
    "conscientiousness": [("burstiness", -1), ("failure_rate", -1), ("essential_share", 1), ("payday_share", -1)],
    "extraversion": [("social_share", 1), ("weekend_share", 1), ("n_cities", 1)],
    "agreeableness": [("avg_rating", 1), ("merchant_repeat_ratio", 1), ("notes_share", -1)],
    "neuroticism": [("refund_rate", 1), ("rating_std", 1), ("cv_net_amount", 1)],
}


def _zscore(series: pd.Series) -> pd.Series:
    filled = series.fillna(series.median())
    std = filled.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (filled - filled.mean()) / std


def _cronbach_alpha(item_scores: pd.DataFrame) -> float:
    """Cronbach's alpha over sign-adjusted, standardized indicators."""
    k = item_scores.shape[1]
    if k < 2:
        return float("nan")
    item_variances = item_scores.var(axis=0, ddof=1).sum()
    total_variance = item_scores.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_variances / total_variance))


def add_personality_traits(user_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = user_features.copy()
    meta = {}

    for trait, indicators in TRAITS.items():
        signed_z = pd.DataFrame(index=df.index)
        for col, sign in indicators:
            signed_z[col] = _zscore(df[col]) * sign

        trait_score = signed_z.mean(axis=1)
        df[f"{trait}_raw"] = trait_score
        df[f"{trait}_score"] = trait_score.rank(pct=True) * 100

        alpha = _cronbach_alpha(signed_z)
        corr = signed_z.corr()
        n = corr.shape[0]
        mean_inter_corr = float(
            (corr.to_numpy().sum() - n) / (n * (n - 1))
        ) if n > 1 else float("nan")

        meta[trait] = {
            "indicators": [c for c, _ in indicators],
            "cronbach_alpha": None if np.isnan(alpha) else round(alpha, 4),
            "mean_inter_indicator_correlation": None if np.isnan(mean_inter_corr) else round(mean_inter_corr, 4),
            "reliable": bool(alpha >= 0.5) if not np.isnan(alpha) else False,
        }

    return df, meta
