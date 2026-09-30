"""Parse user_agent (unique values only, for speed) into OS/device columns."""
import re

import pandas as pd


def _parse_one(user_agent: str, thresholds: dict) -> dict:
    if pd.isna(user_agent):
        return {"os_family": None, "device_type": None, "os_major": None,
                "os_version": None, "info_legacy_os": False}

    ua = user_agent.strip()

    if ua == "iPhone":
        return {"os_family": "iOS", "device_type": "Phone", "os_major": None,
                "os_version": None, "info_legacy_os": False}
    if ua == "iPad":
        return {"os_family": "iOS", "device_type": "Tablet", "os_major": None,
                "os_version": None, "info_legacy_os": False}

    match = re.match(r"^Android\s+([\d.]+)$", ua)
    if match:
        version = match.group(1)
        major = float(version.split(".")[0])
        threshold = thresholds.get("Android")
        legacy = threshold is not None and major < threshold
        return {"os_family": "Android", "device_type": "Phone", "os_major": major,
                "os_version": version, "info_legacy_os": bool(legacy)}

    return {"os_family": "Unknown", "device_type": "Unknown", "os_major": None,
            "os_version": ua, "info_legacy_os": False}


def parse_devices(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    thresholds = cfg["cleaning"]["legacy_os_thresholds"]

    unique_agents = df["user_agent"].dropna().unique()
    parsed = {ua: _parse_one(ua, thresholds) for ua in unique_agents}
    parsed_df = pd.DataFrame.from_dict(parsed, orient="index")
    parsed_df.index.name = "user_agent"

    df = df.merge(parsed_df, on="user_agent", how="left")
    return df
