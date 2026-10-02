"""Select and transform the behavioral features used to form user segments.

Geo, device, payment-method category, and the (partly unreliable) personality scores are
deliberately excluded from clustering inputs: they are used only to *profile* segments
after they are formed, so the clusters reflect behavior, not a label we already know.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_raw_features(user_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """bool -> 0/1 and inf -> NaN -> median, without scaling or log-transforming.

    Used both as the base for the scaled clustering matrix and for the raw (interpretable)
    per-segment means reported in segment_profiles.json, so an inf in e.g. lebaran_lift
    (a user with no non-Lebaran activity) never leaks into a reported average.
    """
    X = user_features[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))


def build_feature_matrix(user_features: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, StandardScaler]:
    """Return (scaled feature DataFrame indexed like user_features, fitted scaler)."""
    seg_cfg = cfg["segmentation"]
    feature_cols = seg_cfg["features"]
    log_cols = set(seg_cfg.get("log_features", []))

    X = clean_raw_features(user_features, feature_cols)

    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=user_features.index)

    return X_scaled, scaler
