"""Fit K-Means, select k via silhouette score, persist the model."""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("pipeline")

SILHOUETTE_SAMPLE_SIZE = 5000


def select_k(X: pd.DataFrame, k_min: int, k_max: int, random_state: int) -> dict:
    """Fit K-Means for each k in [k_min, k_max] and score it with silhouette (sampled for speed)."""
    scores = {}
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X))

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
        scores[k] = {"inertia": float(km.inertia_), "silhouette": float(score)}
        logger.info("k=%d -> inertia=%.1f, silhouette=%.4f", k, km.inertia_, score)

    return scores


def fit_final_model(X: pd.DataFrame, k: int, random_state: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    return km


def save_model(model: KMeans, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path) -> KMeans:
    return joblib.load(path)
