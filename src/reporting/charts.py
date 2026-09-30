"""matplotlib charts (PNG), saved into outputs/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, figures_dir: Path, name: str) -> str:
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def daily_volume_chart(transactions: pd.DataFrame, cfg: dict, figures_dir: Path) -> str:
    daily = transactions.groupby(transactions["transaction_date"].dt.date).size()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(daily.index, daily.values, color="#2563eb", linewidth=1.2)
    lebaran_start = pd.Timestamp(cfg["calendar"]["lebaran_start"]).date()
    lebaran_end = pd.Timestamp(cfg["calendar"]["lebaran_end"]).date()
    ax.axvspan(lebaran_start, lebaran_end, color="#f59e0b", alpha=0.25, label="Eid al-Fitr period")
    ax.set_title("Daily transaction volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    return _save(fig, figures_dir, "daily_volume")


def region_map_chart(transactions: pd.DataFrame, figures_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    sample = transactions.dropna(subset=["lat", "lon"]).sample(
        n=min(20000, len(transactions)), random_state=42
    )
    cities = sample["city"].astype("category")
    scatter = ax.scatter(sample["lon"], sample["lat"], c=cities.cat.codes, cmap="tab10", s=3, alpha=0.5)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)), label=cat)
        for i, cat in enumerate(cities.cat.categories)
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower left", ncol=1)
    ax.set_title("Transaction points by regency/city")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return _save(fig, figures_dir, "region_map")


def bar_chart(categories, values, title, xlabel, ylabel, name, figures_dir: Path, horizontal=False) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    if horizontal:
        ax.barh(categories, values, color="#2563eb")
        ax.invert_yaxis()
    else:
        ax.bar(categories, values, color="#2563eb")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save(fig, figures_dir, name)


def grouped_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, name: str, figures_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    df.plot(x=x_col, y=y_cols, kind="bar", ax=ax, color=["#2563eb", "#f59e0b", "#10b981"][: len(y_cols)])
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, figures_dir, name)


def line_chart(x, series: dict, title, xlabel, ylabel, name, figures_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in series.items():
        ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend()
    return _save(fig, figures_dir, name)


def histogram(values, title, xlabel, name, figures_dir: Path, bins=30) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#2563eb", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    return _save(fig, figures_dir, name)
