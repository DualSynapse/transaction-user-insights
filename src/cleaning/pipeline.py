"""run_cleaning(cfg): combine flags, save data and report."""
import logging

import pandas as pd

from src.cleaning.audit import audit_missing
from src.cleaning.consistency import check_consistency
from src.cleaning.device_parser import parse_devices
from src.cleaning.geo_enrichment import enrich_geo
from src.cleaning.standardize import standardize
from src.utils.io import write_json, write_table

logger = logging.getLogger("pipeline")


def run_cleaning(cfg) -> pd.DataFrame:
    raw_path = cfg.path("raw_csv")
    sep = cfg["cleaning"]["csv_sep"]

    logger.info("Reading raw transactions from %s", raw_path)
    df_raw_for_audit = pd.read_csv(raw_path, sep=sep, dtype=str, keep_default_na=False, na_values=[])

    logger.info("Running missing-value audit")
    missing_audit = audit_missing(df_raw_for_audit, cfg["cleaning"]["missing_tokens"])

    df_raw = df_raw_for_audit.replace({"": pd.NA})

    logger.info("Standardizing types and categories")
    df = standardize(df_raw, cfg.raw)

    logger.info("Checking consistency and computing dq_* flags")
    df = check_consistency(df, cfg.raw)

    logger.info("Enriching with geographic data")
    df = enrich_geo(df, cfg)

    logger.info("Parsing device/user-agent information")
    df = parse_devices(df, cfg.raw)

    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    df["dq_any_issue"] = df[dq_cols].any(axis=1)

    out_path = cfg.path("transactions_clean")
    write_table(df, out_path)
    logger.info("Wrote cleaned transactions to %s (%d rows, %d cols)", out_path, *df.shape)

    dq_report = _build_dq_report(df, missing_audit, dq_cols)
    write_json(dq_report, cfg.path("dq_report"))
    logger.info("Wrote data quality report to %s", cfg.path("dq_report"))

    return df


def _build_dq_report(df: pd.DataFrame, missing_audit: dict, dq_cols: list[str]) -> dict:
    return {
        "n_transactions": len(df),
        "n_users": int(df["user_id"].nunique()),
        "n_merchants": int(df["merchant_id"].nunique()),
        "n_mcc_categories": int(df["merchant_category_id"].nunique()),
        "date_range": [str(df["transaction_date"].min()), str(df["transaction_date"].max())],
        "missing_audit": missing_audit,
        "transaction_status_breakdown": df["transaction_status_clean"].value_counts().to_dict(),
        "dq_flag_counts": {c: int(df[c].sum()) for c in dq_cols},
        "n_any_dq_issue": int(df["dq_any_issue"].sum()),
        "geo_match_method_counts": df["geo_match_method"].value_counts().to_dict(),
        "geo_match_coverage_pct": round(float((df["geo_match_method"] != "unmatched").mean()) * 100, 2),
        "province_regency_counts": [
            {"province": p, "city": c, "count": int(n)}
            for (p, c), n in df.groupby(["province", "city"]).size().sort_values(ascending=False).items()
        ],
    }
