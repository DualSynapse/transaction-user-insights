import numpy as np
import pandas as pd
import pytest

from src.features.demographics import _confidence
from src.features.personality import _cronbach_alpha, _zscore, add_personality_traits
from src.features.user_level import _burstiness, _entropy


def test_entropy_uniform_vs_concentrated():
    uniform = pd.Series([10, 10, 10, 10])
    concentrated = pd.Series([100, 1, 1, 1])
    assert _entropy(uniform) > _entropy(concentrated)
    assert _entropy(pd.Series([5])) == 0.0


def test_burstiness_regular_vs_bursty():
    regular = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bursty = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
    assert _burstiness(regular) == pytest.approx(-1.0, abs=0.05)
    assert _burstiness(bursty) > 0
    assert _burstiness(np.array([1.0])) == 0.0


def test_confidence_levels():
    count = pd.Series([0, 1, 3, 10])
    expected = pd.Series([1.0, 1.0, 1.0, 1.0])
    conf = _confidence(count, expected)
    assert conf.tolist() == ["none", "low", "medium", "high"]


def test_cronbach_alpha_high_for_correlated_items():
    n = 200
    base = np.random.RandomState(0).normal(size=n)
    items = pd.DataFrame({
        "a": base + np.random.RandomState(1).normal(scale=0.05, size=n),
        "b": base + np.random.RandomState(2).normal(scale=0.05, size=n),
        "c": base + np.random.RandomState(3).normal(scale=0.05, size=n),
    })
    alpha = _cronbach_alpha(items)
    assert alpha > 0.9


def test_cronbach_alpha_low_for_independent_items():
    rng = np.random.RandomState(0)
    items = pd.DataFrame({
        "a": rng.normal(size=500),
        "b": rng.normal(size=500),
        "c": rng.normal(size=500),
    })
    alpha = _cronbach_alpha(items)
    assert alpha < 0.3


def test_add_personality_traits_produces_percentiles_0_100():
    n = 50
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "category_entropy": rng.random(n),
        "n_categories": rng.randint(1, 10, n),
        "merchant_diversity": rng.random(n),
        "burstiness": rng.uniform(-1, 1, n),
        "failure_rate": rng.random(n),
        "essential_share": rng.random(n),
        "payday_share": rng.random(n),
        "social_share": rng.random(n),
        "weekend_share": rng.random(n),
        "n_cities": rng.randint(1, 3, n),
        "avg_rating": rng.uniform(1, 5, n),
        "merchant_repeat_ratio": rng.random(n),
        "notes_share": rng.random(n),
        "refund_rate": rng.random(n),
        "rating_std": rng.random(n),
        "cv_net_amount": rng.random(n),
    })
    out, meta = add_personality_traits(df)
    assert out["openness_score"].between(0, 100).all()
    assert set(meta.keys()) == {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    assert "cronbach_alpha" in meta["openness"]
