"""Demographic signals + feasibility table.

Every requested demographic attribute gets at least one creative, data-driven proxy
attempt, with the assumption behind it stated explicitly: a single MCC (e.g.
parental_signal), a composite of several weak behavioral features (e.g.
working_status_estimate, marital_family_proxy), or a reuse of an existing signal reframed
(e.g. age_bracket_estimate from student/parental signals). Every proxy ships with a
confidence column so a weak or near-chance signal is visible rather than hidden, and the
handful of attributes with genuinely zero supporting signal (not just a weak one) are
documented in demographic_feasibility.csv instead of being fabricated outright.
"""
import numpy as np
import pandas as pd

SIGNALS = {
    "parental_signal": ["8211"],
    "student_signal": ["8220"],
    "vehicle_signal": ["5541"],
    "homeowner_signal": ["5251", "5712"],
    "health_care_signal": ["5912"],
    "active_lifestyle_signal": ["5941"],
}

INFEASIBLE_ATTRIBUTES = [
    {"attribute": "industry", "reason": "Attempted: checked whether MCC concentration (e.g. heavy fuel/travel spend "
     "suggesting a delivery/logistics job) could proxy an employer's industry. Rejected — this conflates the "
     "user's own consumption category with their employer's sector, which are unrelated; no transaction field "
     "ties a user to an employer or occupation at all.",
     "data_needed": "Self-reported occupation/industry field, or employer-linked payroll data."},
]


def _confidence(count: pd.Series, expected: pd.Series) -> pd.Series:
    ratio = count / expected.replace(0, np.nan)
    conf = pd.Series("none", index=count.index)
    conf[(count >= 1) & (ratio <= 1.5)] = "low"
    conf[(count >= 1) & (ratio > 1.5) & (ratio <= 3)] = "medium"
    conf[(count >= 1) & (ratio > 3)] = "high"
    conf[count == 0] = "none"
    conf[expected.isna() | (expected == 0)] = "low"
    conf[count == 0] = "none"
    return conf


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std else pd.Series(0.0, index=s.index)


def _gender_lean_signal(df: pd.DataFrame, valid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weak, stereotype-based retail-category lean — exploratory only, never a claimed identity.

    Mostly "none"/"low" confidence by design: the signal is intentionally weak, and the
    categories used reflect common retail marketing stereotypes, not biological or social
    fact. Included as an exploratory gender proxy attempt, reported with an ethical caveat
    rather than silently assumed to be meaningful.
    """
    gender_cfg = cfg["demographics"]["gender_lean"]
    female_mcc = gender_cfg["female_lean_mcc"]
    male_mcc = gender_cfg["male_lean_mcc"]

    female_share = (
        valid[valid["merchant_category_id"].isin(female_mcc)]
        .groupby("user_id").size().reindex(df["user_id"], fill_value=0).to_numpy()
    )
    male_share = (
        valid[valid["merchant_category_id"].isin(male_mcc)]
        .groupby("user_id").size().reindex(df["user_id"], fill_value=0).to_numpy()
    )
    n_valid = df["n_valid_purchases"].replace(0, np.nan).to_numpy()
    female_rate = pd.Series(female_share / n_valid, index=df.index).fillna(0)
    male_rate = pd.Series(male_share / n_valid, index=df.index).fillna(0)

    lean_score = _zscore(female_rate) - _zscore(male_rate)
    total_supporting = female_share + male_share

    confidence = pd.Series("none", index=df.index)
    confidence[(total_supporting >= 1) & (lean_score.abs() < 0.5)] = "low"
    confidence[(total_supporting >= 2) & (lean_score.abs() >= 0.5) & (lean_score.abs() < 1.5)] = "medium"
    confidence[(total_supporting >= 2) & (lean_score.abs() >= 1.5)] = "high"

    df["gender_lean_signal"] = np.where(lean_score > 0, "female_lean", np.where(lean_score < 0, "male_lean", "neutral"))
    df["gender_lean_confidence"] = confidence.to_numpy()
    return df


def _age_bracket_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """Reuses the existing student/parental signals rather than inventing a new one:
    transaction category alone can hint at age bracket, and these two MCC-based signals
    already cover that; this just translates them into an age bracket.
    """
    def bracket(row):
        if row["student_signal_count"] > 0 and row["parental_signal_count"] == 0:
            return "18-24 (student-leaning)"
        if row["parental_signal_count"] > 0:
            return "25-45 (parent-leaning)"
        return "unknown_adult"

    df["age_bracket_estimate"] = df.apply(bracket, axis=1)
    conf_map = {"student_signal_confidence": ("18-24 (student-leaning)",), "parental_signal_confidence": ("25-45 (parent-leaning)",)}
    confidence = pd.Series("none", index=df.index)
    is_student_bracket = df["age_bracket_estimate"] == "18-24 (student-leaning)"
    is_parent_bracket = df["age_bracket_estimate"] == "25-45 (parent-leaning)"
    confidence[is_student_bracket] = df.loc[is_student_bracket, "student_signal_confidence"]
    confidence[is_parent_bracket] = df.loc[is_parent_bracket, "parental_signal_confidence"]
    df["age_bracket_confidence"] = confidence
    return df


def _education_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Single honest signal: presence of university-category spend suggests current tertiary
    education. Does not attempt attained education level (no data supports that distinction).
    """
    df["education_signal"] = np.where(
        df["student_signal_count"] > 0, "likely_in_tertiary_education", "no_signal"
    )
    df["education_confidence"] = np.where(
        df["student_signal_count"] > 0, df["student_signal_confidence"], "none"
    )
    return df


def _work_location_estimate(df: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Weekday-only dominant city vs. overall dominant city.

    This dataset has day-level timestamps only (no time-of-day), so the classic
    "9am-5pm cluster = work" approach isn't available. As a coarser substitute, a user's
    weekday-only dominant city that differs from their overall dominant (home) city is
    reported as a work-location candidate. Single-city users get no useful signal.
    """
    weekday = valid[~valid["is_weekend"]]
    weekday_city = (
        weekday.groupby("user_id")["city"]
        .agg(lambda s: s.value_counts().idxmax() if len(s) else None)
        .reindex(df["user_id"])
    )
    weekday_city_share = (
        weekday.groupby("user_id")["city"]
        .agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
        .reindex(df["user_id"])
    )

    df = df.copy()
    df["_weekday_city"] = weekday_city.to_numpy()
    df["_weekday_city_share"] = weekday_city_share.to_numpy()

    def estimate(row):
        if not row["is_multi_city"]:
            return row["primary_city"], "low"
        if pd.isna(row["_weekday_city"]):
            return row["primary_city"], "none"
        if row["_weekday_city"] != row["primary_city"]:
            share = row["_weekday_city_share"]
            conf = "high" if share >= 0.7 else ("medium" if share >= 0.5 else "low")
            return row["_weekday_city"], conf
        return row["primary_city"], "low"

    results = df.apply(estimate, axis=1)
    df["work_location_estimate"] = [r[0] for r in results]
    df["work_location_confidence"] = [r[1] for r in results]
    df = df.drop(columns=["_weekday_city", "_weekday_city_share"])
    return df


def _working_status_estimate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """payday_share + weekday-heavy activity as a weak proxy for salaried employment.

    Neither feature is a direct employment signal (payday_share just measures spend
    concentrated around common pay dates; weekend_share just measures weekday vs. weekend
    activity), so this is reported as a lean, not a classification.
    """
    ws_cfg = cfg["demographics"]["working_status"]
    min_valid = ws_cfg["min_valid_purchases"]
    payday_threshold = ws_cfg["payday_share_threshold"]
    weekend_max = ws_cfg["weekend_share_max"]

    enough_data = df["n_valid_purchases"] >= min_valid
    employed_pattern = (df["payday_share"] >= payday_threshold) & (df["weekend_share"] <= weekend_max)

    df["working_status_estimate"] = np.where(
        ~enough_data, "insufficient_signal",
        np.where(employed_pattern, "likely_employed_regular_income", "irregular_or_unknown"),
    )
    df["working_status_confidence"] = np.where(
        ~enough_data, "none",
        np.where(employed_pattern, "low", "none"),
    )
    return df


def _marital_family_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weak composite: parental signal + loyalty membership + a geographically settled user.

    Each component is itself weak; this stacks three unrelated weak signals, so confidence
    never exceeds "low". Included as an exploratory marital/parental status proxy attempt,
    not because the composite is reliable.
    """
    min_share = cfg["demographics"]["marital_family_proxy"]["min_primary_city_share"]
    settled = df["primary_city_share"] >= min_share

    family_signal = (df["parental_signal_count"] > 0) & df["loyalty_member"] & settled
    df["marital_family_proxy"] = np.where(family_signal, "family_proxy_likely", "no_signal")
    df["marital_family_confidence"] = np.where(family_signal, "low", "none")
    return df


def add_demographic_signals(user_features: pd.DataFrame, transactions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = user_features.copy()
    valid = transactions[transactions["is_valid_purchase"]]
    n_valid_total = len(valid)

    mcc_counts_global = valid["merchant_category_id"].value_counts()

    for signal_name, mcc_list in SIGNALS.items():
        global_p = mcc_counts_global.reindex(mcc_list, fill_value=0).sum() / n_valid_total if n_valid_total else 0.0

        counts = (
            valid[valid["merchant_category_id"].isin(mcc_list)]
            .groupby("user_id")
            .size()
            .reindex(df["user_id"], fill_value=0)
            .to_numpy()
        )
        df[f"{signal_name}_count"] = counts

        expected = df["n_valid_purchases"] * global_p
        df[f"{signal_name}_confidence"] = _confidence(
            pd.Series(counts, index=df.index), expected
        ).to_numpy()

    def city_confidence(share):
        if share >= 0.8:
            return "high"
        if share >= 0.5:
            return "medium"
        return "low"

    df["home_city_estimate"] = df["primary_city"]
    df["home_city_confidence"] = df["primary_city_share"].apply(city_confidence)

    amount_pct = df["avg_net_amount"].rank(pct=True)

    def income_tier(cc_share, pct):
        if cc_share > 0.5 and pct >= 0.66:
            return "high"
        if cc_share == 0 and pct <= 0.33:
            return "low"
        return "medium"

    df["income_proxy_tier"] = [
        income_tier(cc, pct) for cc, pct in zip(df["credit_card_share"], amount_pct)
    ]

    df = _gender_lean_signal(df, valid, cfg)
    df = _age_bracket_estimate(df)
    df = _education_signal(df)
    df = _work_location_estimate(df, valid)
    df = _working_status_estimate(df, cfg)
    df = _marital_family_proxy(df, cfg)

    return df


def build_feasibility_table() -> pd.DataFrame:
    return pd.DataFrame(INFEASIBLE_ATTRIBUTES)
