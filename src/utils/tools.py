"""Read/write Parquet (CSV.gz fallback), JSON."""
import json
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as Parquet; fall back to CSV.gz if pyarrow is unavailable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError):
        fallback = path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False, compression="gzip")
        return fallback


def read_table(path: Path) -> pd.DataFrame:
    """Read a table written by write_table, trying Parquet then CSV.gz fallback."""
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, compression="gzip")
    fallback = path.with_suffix(".csv.gz")
    if fallback.exists():
        return pd.read_csv(fallback, compression="gzip")
    raise FileNotFoundError(f"Neither {path} nor {fallback} exists.")


def table_exists(path: Path) -> bool:
    path = Path(path)
    return path.exists() or path.with_suffix(".csv.gz").exists()


def _jsonable(obj):
    """Recursively convert dict keys to str and leave values for json's default=str fallback."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
