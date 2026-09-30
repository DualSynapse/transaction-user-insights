import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_synthetic_csv(path: Path, n=80):
    import random
    random.seed(0)

    mcc_ids = ["5411", "5812", "5541", "8211", "8220", "5912", "5941"]
    payment_methods = ["balance", "credit_card"]
    user_agents = ["Android 9", "Android 4.4.2", "iPhone", "iPad"]
    coords = ["-6.1862, 106.8347", "-7.2575, 112.7521", "-7.8877, 110.3286"]

    rows = []
    for i in range(n):
        completed = i % 5 != 0
        status = "completed" if completed else ("failed" if i % 10 == 0 else "")
        rows.append({
            "transaction_id": f"tx{i}",
            "user_id": f"U{i % 10:03d}",
            "transaction_amount": str(1000.0 * ((i % 20) + 1)),
            "transaction_date": f"{(i % 28) + 1:02d}/03/2024",
            "merchant_id": f"M{i % 15:03d}",
            "merchant_name": f"Merchant {i % 15}",
            "merchant_category_id": mcc_ids[i % len(mcc_ids)],
            "geo_location": coords[i % len(coords)],
            "payment_method": payment_methods[i % 2],
            "user_agent": user_agents[i % len(user_agents)],
            "loyalty_program": ("yes" if i % 3 == 0 else "no") if status else "",
            "discount_applied": ("yes" if i % 4 == 0 else "no") if status else "",
            "promo_amount": (str(100.0 if i % 4 == 0 else 0.0)) if status else "",
            "transaction_notes": ("yes" if i % 6 == 0 else "no") if status else "",
            "merchant_rating": (str((i % 5) + 1)) if status == "completed" or status == "failed" else "",
            "transaction_status": status,
            "is_refunded": ("yes" if i % 11 == 0 else "no") if status else "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, sep=";", index=False)


@pytest.fixture()
def sample_project(tmp_path):
    raw_csv = tmp_path / "transactions.csv"
    _make_synthetic_csv(raw_csv)

    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["raw_csv"] = str(raw_csv)
    for key in ["interim_dir", "processed_dir", "reports_dir", "figures_dir", "logs_dir"]:
        cfg["paths"][key] = str(tmp_path / key)
    cfg["paths"]["transactions_clean"] = str(tmp_path / "interim_dir" / "transactions_clean.parquet")
    cfg["paths"]["dq_report"] = str(tmp_path / "reports_dir" / "data_quality_report.json")
    cfg["paths"]["transactions_enriched"] = str(tmp_path / "processed_dir" / "transactions_enriched.parquet")
    cfg["paths"]["user_features"] = str(tmp_path / "processed_dir" / "user_features.parquet")
    cfg["paths"]["feature_dictionary"] = str(tmp_path / "processed_dir" / "feature_dictionary.csv")
    cfg["paths"]["demographic_feasibility"] = str(tmp_path / "processed_dir" / "demographic_feasibility.csv")
    cfg["paths"]["feature_meta"] = str(tmp_path / "processed_dir" / "feature_meta.json")
    cfg["paths"]["report_pdf"] = str(tmp_path / "reports_dir" / "transaction_insights_report.pdf")
    cfg["paths"]["analysis_results"] = str(tmp_path / "reports_dir" / "analysis_results.json")
    # reuse the already-cached boundary file so the test doesn't need network access
    cfg["paths"]["external_dir"] = str(PROJECT_ROOT / "data" / "external")
    cfg["analysis"]["min_users_per_merchant"] = 1

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return config_path


def test_main_end_to_end_produces_pdf(sample_project):
    result = subprocess.run(
        [sys.executable, "main.py", "--config", str(sample_project)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with open(sample_project, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pdf_path = Path(cfg["paths"]["report_pdf"])
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

    user_features_path = Path(cfg["paths"]["user_features"])
    assert user_features_path.exists()
