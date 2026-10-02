"""run_segmentation(cfg): build features -> select k -> fit K-Means -> profile segments."""
import logging

from src.segmentation.features import build_feature_matrix, clean_raw_features
from src.segmentation.model import fit_final_model, save_model, select_k
from src.segmentation.profiling import build_segment_profiles, label_segments
from src.utils.tools import read_table, write_json, write_table

logger = logging.getLogger("pipeline")


def run_segmentation(cfg):
    seg_cfg = cfg["segmentation"]

    logger.info("Loading user features")
    user_features = read_table(cfg.path("user_features"))

    logger.info("Building segmentation feature matrix (%d candidate features)", len(seg_cfg["features"]))
    X, _scaler = build_feature_matrix(user_features, cfg.raw)

    logger.info("Selecting k in [%d, %d] via silhouette score", seg_cfg["k_min"], seg_cfg["k_max"])
    k_scores = select_k(X, seg_cfg["k_min"], seg_cfg["k_max"], seg_cfg["random_state"])
    best_k = max(k_scores, key=lambda k: k_scores[k]["silhouette"])
    logger.info("Selected k=%d (silhouette=%.4f)", best_k, k_scores[best_k]["silhouette"])

    model = fit_final_model(X, best_k, seg_cfg["random_state"])
    save_model(model, cfg.path("segment_model"))

    user_features = user_features.copy()
    user_features["segment_id"] = model.labels_

    # Raw (unscaled, inf-cleaned) feature values, used for human-readable cluster means —
    # the scaled/log-transformed X above is only for fitting the model.
    raw_features = clean_raw_features(user_features, seg_cfg["features"])
    for col in seg_cfg["features"]:
        user_features[col] = raw_features[col]

    cluster_means = user_features.groupby("segment_id")[seg_cfg["features"]].mean()
    labels = label_segments(cluster_means)
    user_features["segment_label"] = user_features["segment_id"].map(labels)

    write_table(user_features[["user_id", "segment_id", "segment_label"]], cfg.path("user_segments"))
    logger.info("Wrote user segments to %s", cfg.path("user_segments"))

    profiles = build_segment_profiles(user_features, seg_cfg["features"], seg_cfg["profiling_features"])
    output = {
        "k_selection": {str(k): v for k, v in k_scores.items()},
        "selected_k": best_k,
        "segments": profiles,
    }
    write_json(output, cfg.path("segment_profiles"))
    logger.info("Wrote segment profiles to %s", cfg.path("segment_profiles"))

    return user_features
