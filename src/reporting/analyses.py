"""Data audit + 8 business questions + statistical tests."""
import numpy as np
import pandas as pd
from scipy import stats

from src.reporting import charts


def _chi2(table: pd.DataFrame) -> dict:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else float("nan")
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": cramers_v}


def data_audit(transactions: pd.DataFrame, dq_report: dict) -> dict:
    return {
        "n_transactions": len(transactions),
        "n_users": int(transactions["user_id"].nunique()),
        "n_merchants": int(transactions["merchant_id"].nunique()),
        "n_mcc_categories": int(transactions["merchant_category_id"].nunique()),
        "date_range": dq_report["date_range"],
        "status_breakdown": dq_report["transaction_status_breakdown"],
        "dq_flag_counts": dq_report["dq_flag_counts"],
        "geo_match_coverage_pct": dq_report["geo_match_coverage_pct"],
        "limitations": [
            "Data is synthetic: 93,584 transactions (33.6%) have a blank status (neither completed "
            "nor failed) and correspondingly blank rating/refund/discount/notes/loyalty fields; these "
            "are treated as a third 'pending' category rather than as errors.",
            "100% of transactions with a rating are on failed transactions (9,393 of 9,393 failed "
            "transactions carry a rating) — the opposite of what would be expected operationally, "
            "and a sign the rating and status fields were generated independently.",
            "The administrative boundary file (GADM, 394 regencies/cities) predates several splits; "
            "there are 514 regencies/cities today. This has no effect on the 9 cities present in this "
            "dataset.",
            "No promo-source column exists, so merchant- vs platform-funded discounts are inferred "
            "heuristically from each merchant's discount rate vs. the platform baseline.",
        ],
    }


def q1_repeat_retention_by_region(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    by_city = valid.groupby("city")["user_id"].agg(n_transactions="count", n_users="nunique")
    repeat_users = valid.groupby(["city", "user_id"]).size().reset_index(name="n")
    repeat_rate = (
        repeat_users.assign(is_repeat=repeat_users["n"] > 1)
        .groupby("city")["is_repeat"].mean()
    )
    by_city["repeat_rate"] = repeat_rate

    march_users = set(valid.loc[valid["transaction_date"].dt.month == 3, "user_id"])
    later_users = set(valid.loc[valid["transaction_date"].dt.month.isin([4, 5]), "user_id"])
    user_city = valid.groupby("user_id")["city"].agg(lambda s: s.value_counts().idxmax())
    retained = pd.Series({u: u in later_users for u in march_users})
    retention_by_city = retained.groupby(user_city.reindex(retained.index)).mean()

    contingency = pd.crosstab(
        repeat_users["city"], repeat_users["n"] > 1
    )
    test = _chi2(contingency)

    chart = charts.bar_chart(
        by_city.index, by_city["repeat_rate"] * 100,
        "Repeat purchase rate by city", "City", "Repeat rate (%)",
        "q1_repeat_rate_by_city", figures_dir, horizontal=True,
    )

    return {
        "table": by_city.reset_index().to_dict(orient="records"),
        "march_cohort_retention_by_city": retention_by_city.round(4).to_dict(),
        "chi2_test": test,
        "chart": chart,
        "significant": test["p_value"] < cfg["analysis"]["alpha"],
    }


def q2_merchant_repeat_and_promo(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    min_users = cfg["analysis"]["min_users_per_merchant"]

    merchant_users = valid.groupby("merchant_id")["user_id"].agg(n_users="nunique", n_transactions="count")
    eligible = merchant_users[merchant_users["n_users"] >= min_users].index

    repeat = valid[valid["merchant_id"].isin(eligible)].groupby(["merchant_id", "user_id"]).size().reset_index(name="n")
    repeat_rate = repeat.assign(is_repeat=repeat["n"] > 1).groupby("merchant_id")["is_repeat"].mean()

    platform_discount_rate = valid["discount_applied"].fillna(False).mean()
    merchant_discount = valid.groupby("merchant_id")["discount_applied"].apply(lambda s: s.fillna(False).mean())
    merchant_n = valid.groupby("merchant_id").size()

    def binom_flag(rate, n):
        if n < min_users:
            return None
        successes = int(round(rate * n))
        p = stats.binomtest(successes, n, platform_discount_rate).pvalue
        return "merchant_funded" if (p < cfg["analysis"]["alpha"] and rate > platform_discount_rate) else "platform_baseline"

    promo_source_heuristic = pd.Series(
        {m: binom_flag(merchant_discount[m], merchant_n[m]) for m in eligible}
    )

    top_repeat = repeat_rate.sort_values(ascending=False).head(15)
    chart = charts.bar_chart(
        top_repeat.index, top_repeat.values * 100,
        f"Top merchant repeat-user rate (>= {min_users} users)", "Merchant ID", "Repeat rate (%)",
        "q2_merchant_repeat_rate", figures_dir, horizontal=True,
    )

    return {
        "n_eligible_merchants": len(eligible),
        "platform_discount_rate": round(float(platform_discount_rate), 4),
        "promo_source_counts": promo_source_heuristic.value_counts(dropna=False).to_dict(),
        "top_repeat_merchants": top_repeat.round(4).to_dict(),
        "chart": chart,
        "note": "No promo-source column exists; 'merchant_funded' vs 'platform_baseline' is a heuristic "
                "flag from a one-sided binomial test of each merchant's discount rate against the "
                "platform-wide rate, not a ground-truth label.",
    }


def q3_category_performance(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    agg = valid.groupby("mcc_group").agg(
        n_transactions=("transaction_id", "count"),
        gmv=("net_amount", "sum"),
        n_users=("user_id", "nunique"),
    ).sort_values("n_transactions", ascending=False)

    march = valid[valid["transaction_date"].dt.month == 3].groupby("mcc_group").size()
    may = valid[valid["transaction_date"].dt.month == 5].groupby("mcc_group").size()
    growth = ((may - march) / march.replace(0, np.nan) * 100).round(1)
    agg["march_to_may_growth_pct"] = growth

    lebaran_daily = valid[valid["is_lebaran_period"]].groupby("mcc_group").size() / 7
    non_lebaran_days = (valid["transaction_date"].max() - valid["transaction_date"].min()).days + 1 - 7
    non_lebaran_daily = valid[~valid["is_lebaran_period"]].groupby("mcc_group").size() / non_lebaran_days
    agg["lebaran_lift"] = (lebaran_daily / non_lebaran_daily).round(2)

    chart = charts.bar_chart(
        agg.index, agg["n_transactions"],
        "Transactions by category group", "Category", "Transactions",
        "q3_category_volume", figures_dir, horizontal=True,
    )

    return {
        "table": agg.reset_index().round(2).to_dict(orient="records"),
        "best_selling": agg["n_transactions"].idxmax(),
        "rarely_ordered": agg["n_transactions"].idxmin(),
        "chart": chart,
    }


def q4_devices_and_os(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    by_os = valid.groupby("os_family").agg(
        n_users=("user_id", "nunique"), gmv=("net_amount", "sum")
    )
    fail_by_os = transactions.groupby("os_family")["is_failed"].mean()
    by_os["failure_rate"] = fail_by_os

    android = transactions[transactions["os_family"] == "Android"].dropna(subset=["os_major"])
    coverage = (
        android.groupby("os_major")["user_id"].nunique().sort_index(ascending=False).cumsum()
        / android["user_id"].nunique()
    )
    target = cfg["analysis"]["os_coverage_target"]
    candidates = coverage[coverage >= target]
    recommended_min_version = float(candidates.index.min()) if len(candidates) else float(coverage.index.min())

    chart = charts.line_chart(
        coverage.index.astype(str), {"Cumulative user coverage": coverage.values},
        "Android OS version coverage (cumulative, newest first)", "Android major version", "User coverage",
        "q4_os_coverage", figures_dir,
    )

    return {
        "by_os": by_os.reset_index().round(4).to_dict(orient="records"),
        "recommended_min_android_version": recommended_min_version,
        "coverage_target": target,
        "chart": chart,
    }


def q5_balance_vs_credit_card(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    by_method = valid.groupby("payment_method").agg(
        n_transactions=("transaction_id", "count"),
        aov=("net_amount", "mean"),
    )
    refund_rate = transactions.groupby("payment_method").apply(
        lambda g: (g["is_refunded"].fillna(False) & g["is_completed"]).sum() / max(g["is_completed"].sum(), 1)
    )
    fail_rate = transactions.groupby("payment_method")["is_failed"].mean()
    discount_rate = valid.groupby("payment_method")["discount_applied"].apply(lambda s: s.fillna(False).mean())
    by_method["refund_rate"] = refund_rate
    by_method["failure_rate"] = fail_rate
    by_method["discount_rate"] = discount_rate

    mdr = cfg["assumptions"]["mdr"]
    by_method["mdr_cost"] = [
        by_method.loc[m, "aov"] * by_method.loc[m, "n_transactions"] * mdr.get(m, 0) for m in by_method.index
    ]

    x = valid.loc[valid["payment_method"] == "balance", "net_amount"]
    y = valid.loc[valid["payment_method"] == "credit_card", "net_amount"]
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")

    chart = charts.bar_chart(
        by_method.index, by_method["aov"],
        "Average order value by payment method", "Payment method", "AOV",
        "q5_aov_by_method", figures_dir,
    )

    return {
        "table": by_method.reset_index().round(2).to_dict(orient="records"),
        "mann_whitney_u": {"u_stat": float(u_stat), "p_value": float(p_value)},
        "significant": p_value < cfg["analysis"]["alpha"],
        "total_mdr_cost": round(float(by_method["mdr_cost"].sum()), 2),
        "chart": chart,
    }


def q6_rating_drivers(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    completed = transactions[transactions["is_completed"]]
    ratings = completed.dropna(subset=["merchant_rating_clean"])

    drivers = {}
    for factor in ["payment_method", "mcc_group", "discount_applied"]:
        means = ratings.groupby(factor)["merchant_rating_clean"].mean().round(3)
        drivers[factor] = means.to_dict()

    refund_notes = pd.crosstab(ratings["is_refunded"].fillna(False), ratings["transaction_notes"].fillna(False))
    test = _chi2(refund_notes) if refund_notes.shape[0] > 1 and refund_notes.shape[1] > 1 else None

    chart = charts.bar_chart(
        list(drivers["mcc_group"].keys()), list(drivers["mcc_group"].values()),
        "Mean rating by category", "Category", "Mean rating",
        "q6_rating_by_category", figures_dir, horizontal=True,
    )

    return {
        "mean_rating_by_factor": drivers,
        "refund_notes_crosstab": refund_notes.to_dict(),
        "chi2_test": test,
        "chart": chart,
    }


def q7_loyalty_and_devices(transactions: pd.DataFrame, user_features: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    members = user_features[user_features["loyalty_member"]]
    non_members = user_features[~user_features["loyalty_member"]]

    def summarize(g):
        return {
            "n_users": len(g),
            "avg_n_transactions": round(float(g["n_transactions"].mean()), 2),
            "avg_total_net_amount": round(float(g["total_net_amount"].mean()), 2),
            "avg_active_months": round(float(g["active_months"].mean()), 2),
        }

    comparison = {"members": summarize(members), "non_members": summarize(non_members)}

    u_stat, p_value = stats.mannwhitneyu(
        members["n_transactions"], non_members["n_transactions"], alternative="two-sided"
    )

    device_of_engaged = (
        user_features.sort_values("n_transactions", ascending=False)
        .head(int(len(user_features) * 0.1))["os_family"].value_counts(normalize=True)
    )

    chart = charts.bar_chart(
        ["members", "non_members"],
        [comparison["members"]["avg_n_transactions"], comparison["non_members"]["avg_n_transactions"]],
        "Average transactions: loyalty members vs. non-members", "Group", "Avg. transactions",
        "q7_loyalty_comparison", figures_dir,
    )

    return {
        "comparison": comparison,
        "mann_whitney_u": {"u_stat": float(u_stat), "p_value": float(p_value)},
        "significant": p_value < cfg["analysis"]["alpha"],
        "device_share_top_decile_engaged_users": device_of_engaged.round(4).to_dict(),
        "chart": chart,
    }


def q8_regions_without_promo(transactions: pd.DataFrame, cfg: dict, figures_dir) -> dict:
    valid = transactions[transactions["is_valid_purchase"]]
    user_used_promo = valid.groupby("user_id")["discount_applied"].apply(lambda s: s.fillna(False).any())
    user_city = valid.groupby("user_id")["city"].agg(lambda s: s.value_counts().idxmax())

    never_promo = (~user_used_promo).groupby(user_city).mean()

    contingency = pd.crosstab(user_city, user_used_promo)
    test = _chi2(contingency)

    chart = charts.bar_chart(
        never_promo.index, never_promo.values * 100,
        "Share of users who never used a promo, by city", "City", "Never-promo users (%)",
        "q8_never_promo_by_city", figures_dir, horizontal=True,
    )

    return {
        "never_promo_share_by_city": never_promo.round(4).to_dict(),
        "chi2_test": test,
        "significant": test["p_value"] < cfg["analysis"]["alpha"],
        "chart": chart,
        "interpretation_note": "A higher never-promo share can reflect promo unavailability, lower price "
                                "sensitivity, or lower promo awareness in that city; this dataset cannot "
                                "distinguish between the three without promo-eligibility or exposure data.",
    }


def run_all_analyses(transactions: pd.DataFrame, user_features: pd.DataFrame, dq_report: dict, cfg: dict, figures_dir) -> dict:
    return {
        "audit": data_audit(transactions, dq_report),
        "daily_volume_chart": charts.daily_volume_chart(transactions, cfg, figures_dir),
        "region_map_chart": charts.region_map_chart(transactions, figures_dir),
        "q1": q1_repeat_retention_by_region(transactions, cfg, figures_dir),
        "q2": q2_merchant_repeat_and_promo(transactions, cfg, figures_dir),
        "q3": q3_category_performance(transactions, cfg, figures_dir),
        "q4": q4_devices_and_os(transactions, cfg, figures_dir),
        "q5": q5_balance_vs_credit_card(transactions, cfg, figures_dir),
        "q6": q6_rating_drivers(transactions, cfg, figures_dir),
        "q7": q7_loyalty_and_devices(transactions, user_features, cfg, figures_dir),
        "q8": q8_regions_without_promo(transactions, cfg, figures_dir),
    }
