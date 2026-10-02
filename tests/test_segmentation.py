import numpy as np
import pandas as pd
import pytest

from src.segmentation.features import build_feature_matrix, clean_raw_features
from src.segmentation.model import fit_final_model, select_k
from src.segmentation.profiling import build_segment_profiles, label_segments

SEG_CFG = {
    "segmentation": {
        "random_state": 0,
        "k_min": 2,
        "k_max": 3,
        "log_features": ["avg_net_amount"],
        "features": [
            "recency_days", "active_months", "n_valid_purchases", "avg_net_amount",
            "cv_net_amount", "category_entropy", "merchant_diversity", "merchant_repeat_ratio",
            "discount_tx_share", "credit_card_share", "failure_rate", "refund_rate",
            "weekend_share", "lebaran_lift", "loyalty_member",
        ],
        "profiling_features": ["primary_city", "avg_rating"],
    }
}


def _synthetic_user_features(n=60, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "recency_days": rng.uniform(0, 30, n),
        "active_months": rng.uniform(1, 3, n),
        "n_valid_purchases": rng.randint(0, 20, n),
        "avg_net_amount": rng.uniform(0, 100000, n),
        "cv_net_amount": rng.uniform(0, 2, n),
        "category_entropy": rng.uniform(0, 2, n),
        "merchant_diversity": rng.uniform(0, 2, n),
        "merchant_repeat_ratio": rng.uniform(0, 1, n),
        "discount_tx_share": rng.uniform(0, 1, n),
        "credit_card_share": rng.uniform(0, 1, n),
        "failure_rate": rng.uniform(0, 0.2, n),
        "refund_rate": rng.uniform(0, 0.2, n),
        "weekend_share": rng.uniform(0, 1, n),
        "lebaran_lift": rng.choice([0.5, 1.0, 2.0, np.inf], n),
        "loyalty_member": rng.choice([True, False], n),
        "primary_city": rng.choice(["City A", "City B"], n),
        "avg_rating": rng.uniform(1, 5, n),
    })
    return df


def test_clean_raw_features_removes_inf_and_casts_bool():
    df = _synthetic_user_features()
    cleaned = clean_raw_features(df, SEG_CFG["segmentation"]["features"])

    assert not np.isinf(cleaned["lebaran_lift"]).any()
    assert cleaned["loyalty_member"].dtype == float
    assert cleaned.isna().sum().sum() == 0


def test_build_feature_matrix_is_scaled():
    df = _synthetic_user_features()
    X, scaler = build_feature_matrix(df, SEG_CFG)

    assert list(X.columns) == SEG_CFG["segmentation"]["features"]
    assert X.shape[0] == len(df)
    # standardized columns should have ~mean 0
    assert X.mean().abs().max() < 1e-6


def test_select_k_and_fit_final_model():
    df = _synthetic_user_features(n=60)
    X, _ = build_feature_matrix(df, SEG_CFG)

    scores = select_k(X, k_min=2, k_max=3, random_state=0)
    assert set(scores.keys()) == {2, 3}
    for v in scores.values():
        assert -1.0 <= v["silhouette"] <= 1.0

    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    model = fit_final_model(X, best_k, random_state=0)
    assert len(model.labels_) == len(df)
    assert len(set(model.labels_)) == best_k


def test_label_segments_assigns_distinct_names():
    cluster_means = pd.DataFrame({
        "recency_days": [2, 50, 10],
        "active_months": [3, 1, 2],
        "n_valid_purchases": [20, 1, 10],
        "avg_net_amount": [100000, 5000, 50000],
        "discount_tx_share": [0.1, 0.1, 0.8],
    }, index=[0, 1, 2])

    labels = label_segments(cluster_means)

    assert len(set(labels.values())) == 3
    assert labels[0] == "Champions (High-Value Loyal)"
    assert labels[1] == "At-Risk / Dormant"
    assert labels[2] == "Bargain Hunters"


def test_build_segment_profiles_structure():
    df = _synthetic_user_features(n=30)
    df["segment_id"] = [i % 3 for i in range(30)]
    df["segment_label"] = df["segment_id"].map({0: "A", 1: "B", 2: "C"})

    profiles = build_segment_profiles(df, SEG_CFG["segmentation"]["features"], SEG_CFG["segmentation"]["profiling_features"])

    assert set(profiles.keys()) == {"0", "1", "2"}
    for p in profiles.values():
        assert p["n_users"] == 10
        assert pytest.approx(p["share_of_users"], abs=0.01) == 10 / 30
        assert "avg_rating" in p["profiling_means"]
        assert "primary_city" in p["top_categories"]
