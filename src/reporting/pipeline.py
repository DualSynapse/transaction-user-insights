"""run_report(cfg): analyses -> charts -> PDF."""
import logging

from src.reporting.analyses import run_all_analyses
from src.reporting.pdf_builder import ReportBuilder
from src.utils.tools import read_json, read_table, write_json

logger = logging.getLogger("pipeline")


def _sig_text(significant: bool, p_value: float, alpha: float) -> str:
    if significant:
        return f"statistically significant (p = {p_value:.4f} < alpha = {alpha})"
    return f"not statistically significant (p = {p_value:.4f} >= alpha = {alpha})"


def _fmt_pct_dict(d: dict) -> str:
    return ", ".join(f"{k} {v*100:.1f}%" for k, v in d.items())


def _fmt_rate_n_dict(d: dict) -> str:
    return ", ".join(f"{k} {v['rate']*100:.1f}% (n={v['n']:,})" for k, v in d.items())


def _fmt_rating_n_dict(d: dict) -> str:
    return ", ".join(f"{k} {v['mean_rating']:.2f} (n={v['n']:,})" for k, v in d.items())


def _executive_summary(results: dict, cfg: dict) -> list[str]:
    audit = results["audit"]
    seg = results["segmentation"]
    q3 = results["q3"]
    q4 = results["q4"]
    q7 = results["q7"]
    largest_segment = max(seg["segments"].values(), key=lambda s: s["n_users"])
    findings = [
        f"{audit['n_transactions']:,} transactions from {audit['n_users']:,} users across "
        f"{audit['n_merchants']:,} merchants were processed, covering {audit['date_range'][0][:10]} "
        f"to {audit['date_range'][1][:10]}.",
        f"Users split into {seg['selected_k']} behavioral segments; the largest is "
        f"'{largest_segment['label']}' at {largest_segment['share_of_users']*100:.1f}% of users. "
        f"Chapter 4 profiles every segment and Chapter 6.3 breaks down each supplementary business question by segment.",
        f"'{q3['best_selling']}' is the best-selling category by transaction count; "
        f"'{q3['rarely_ordered']}' is the least ordered.",
        f"The recommended minimum Android version to cover {int(q4['coverage_target']*100)}% of "
        f"active Android users is {q4['recommended_min_android_version']}.",
        f"Loyalty members transact {q7['comparison']['members']['avg_n_transactions']} times on "
        f"average vs. {q7['comparison']['non_members']['avg_n_transactions']} for non-members "
        f"({_sig_text(q7['significant'], q7['mann_whitney_u']['p_value'], cfg['analysis']['alpha'])}).",
        f"Transaction volume rises sharply during the Eid al-Fitr window ({cfg['calendar']['lebaran_start']} "
        f"to {cfg['calendar']['lebaran_end']}), the strongest single pattern in the data.",
        "This dataset is synthetic: most cross-variable relationships tested are weak or not "
        "statistically significant. Chapter 5 (assumptions & limitations) covers this in detail.",
    ]
    return findings


def _build_pdf(results: dict, feature_meta: dict, cfg, figures_dir) -> None:
    audit = results["audit"]
    period = f"{audit['date_range'][0][:10]} to {audit['date_range'][1][:10]}"
    builder = ReportBuilder(cfg.path("report_pdf"), "Transaction User Insights Report", period)

    # --- Core deliverable: feature engineering methodology + user segmentation ---
    builder.h1("1. Executive Summary")
    builder.bullets(_executive_summary(results, cfg.raw))
    builder.page_break()

    builder.h1("2. Data Audit")
    builder.p(
        f"The dataset contains {audit['n_transactions']:,} transactions, {audit['n_users']:,} users, "
        f"{audit['n_merchants']:,} merchants and {audit['n_mcc_categories']} MCC categories. This chapter "
        f"is kept brief; it exists to establish how much to trust the features built in Chapter 3."
    )
    builder.h2("Transaction status breakdown")
    builder.table(["Status", "Count"], [[k, f"{v:,}"] for k, v in audit["status_breakdown"].items()])
    builder.h2("Key limitations")
    builder.bullets(audit["limitations"])
    builder.page_break()

    builder.h1("3. Feature Engineering: Methodology, Assumptions & Findings")
    _add_feature_methodology(builder, results["feature_methodology"], feature_meta)

    builder.h1("4. User Segmentation: Methodology & Segment Stories")
    _add_segmentation(builder, results["segmentation"], results["feature_methodology"]["demographic_global_rates"])

    builder.h1("5. Assumptions & Limitations")
    _add_assumptions(builder, cfg.raw)
    builder.page_break()

    # --- Supplementary material: additional exploration beyond the core deliverables ---
    builder.h1("6. Appendix: Supplementary Analysis")
    builder.p(
        "Everything in this appendix goes beyond the core feature engineering and segmentation "
        "deliverables in the chapters above. It demonstrates additional exploration of the same "
        "data: regional/temporal patterns and eight concrete business questions, each with a "
        "statistical test and a per-segment breakdown using the segmentation from Chapter 4."
    )
    builder.page_break()

    builder.h2("6.1 Regional Profile")
    builder.p(f"Geo-matching coverage: {audit['geo_match_coverage_pct']}% of transactions matched to a regency/city.")
    builder.image(results["region_map_chart"])
    builder.page_break()

    builder.h2("6.2 Time Patterns")
    builder.image(results["daily_volume_chart"])
    builder.p(
        "Volume rises markedly during the Eid al-Fitr period (7-13 April 2024), consistent with "
        "the initial data exploration."
    )
    builder.page_break()

    builder.h2("6.3 Eight Business Questions")
    _add_business_questions(builder, results, cfg.raw)

    builder.build()


def _top_value(top_categories: dict) -> str | None:
    if not top_categories:
        return None
    return max(top_categories.items(), key=lambda kv: kv[1])[0]


OVER_INDEX_LABELS = {
    "likely_in_tertiary_education": "currently-studying signal",
    "likely_employed_regular_income": "regular-income (payday) signal",
    "family_proxy_likely": "family/parental signal",
    "female_lean": "female-leaning retail signal",
    "male_lean": "male-leaning retail signal",
    "18-24 (student-leaning)": "18-24 student-leaning age signal",
    "25-45 (parent-leaning)": "25-45 parent-leaning age signal",
}


def _over_indexed_traits(profile: dict, global_rates: dict, min_factor: float = 1.3, top_n: int = 2) -> list[str]:
    """Which rare demographic signals this segment has MORE of than the overall population.

    Raw top-category would just report the majority "no_signal"/"unknown" value for every
    segment, since these proxies are weak and rare (see Chapter 3.10 coverage figures) — this
    instead surfaces what's actually distinctive about THIS segment, the same "over-index"
    framing retail analytics uses for segment descriptions.
    """
    tc = profile["top_categories"]
    source_cols = {
        "likely_in_tertiary_education": "education_signal",
        "likely_employed_regular_income": "working_status_estimate",
        "family_proxy_likely": "marital_family_proxy",
        "female_lean": "gender_lean_signal",
        "male_lean": "gender_lean_signal",
        "18-24 (student-leaning)": "age_bracket_estimate",
        "25-45 (parent-leaning)": "age_bracket_estimate",
    }
    scored = []
    for value, col in source_cols.items():
        seg_rate = tc.get(col, {}).get(value, 0.0)
        global_rate = global_rates.get(value, 0.0)
        if global_rate > 0 and seg_rate > 0:
            factor = seg_rate / global_rate
            if factor >= min_factor:
                scored.append((factor, f"{OVER_INDEX_LABELS[value]} ({factor:.1f}x the overall rate)"))
    scored.sort(key=lambda t: -t[0])
    return [label for _, label in scored[:top_n]]


def _persona_story(profile: dict, global_rates: dict) -> str:
    """One narrative sentence per segment combining behavior and demographic proxies.

    Turns a segment's raw statistics into a short, readable description (e.g. a city, an
    income tier, a payment pattern, and whichever demographic proxies are over-represented
    in that segment). All demographic terms here are proxies with their own confidence
    levels (see Chapter 3.10) stacked into one sentence; the story is illustrative, not a
    certified fact about any individual.
    """
    tc = profile["top_categories"]
    fm = profile["feature_means"]

    city = _top_value(tc.get("primary_city", {})) or "a mix of cities"
    payment = _top_value(tc.get("payment_profile", {})) or "mixed payment"
    income = _top_value(tc.get("income_proxy_tier", {})) or "unknown income"

    bits = [f"Typically based in {city}", f"{income}-income proxy", f"{payment.replace('_', ' ')} payment"]
    standout = _over_indexed_traits(profile, global_rates)
    if standout:
        bits.append("stands out for: " + ", ".join(standout))
    else:
        bits.append("no demographic proxy is notably over-represented vs. the overall population")
    bits.append(f"averages {fm['n_valid_purchases']:.1f} purchases at {fm['avg_net_amount']:,.0f} each")

    return "; ".join(bits) + "."


def _add_segmentation(builder: ReportBuilder, seg: dict, global_rates: dict) -> None:
    builder.p(
        f"Users were grouped into behavioral segments with K-Means (k={seg['selected_k']}, chosen by "
        f"silhouette score across k=3-6). Clustering used only behavioral features (recency, activity, "
        f"value, diversity, discount/loyalty usage, payment mix, quality); geography, device, payment "
        f"category, personality scores, and the demographic proxies from Chapter 3.10 were deliberately "
        f"excluded from the clustering inputs and used only to write the persona stories below, so a "
        f"segment is not simply a relabeled existing column."
    )
    builder.h2("Why not plain RFM")
    builder.p(
        "This segmentation is deliberately designed to go beyond traditional RFM. The 15 "
        "clustering inputs here include recency/frequency/monetary (recency_days, "
        "n_valid_purchases, avg_net_amount) but also diversity (category_entropy, "
        "merchant_diversity), promo/loyalty behavior (discount_tx_share, loyalty_member), "
        "payment mix (credit_card_share), quality (failure_rate, refund_rate), and time "
        "patterns (weekend_share, lebaran_lift), so segments can separate on behavior that "
        "RFM alone would miss entirely."
    )
    builder.h2("k selection (silhouette score)")
    builder.table(
        ["k", "Inertia", "Silhouette"],
        [[k, round(v["inertia"], 1), round(v["silhouette"], 4)] for k, v in seg["k_selection"].items()],
    )

    builder.h2("Segment sizes and value")
    builder.image(seg["size_chart"])
    builder.image(seg["value_chart"])

    builder.h2("Segment profiles & stories")
    for seg_id, profile in sorted(seg["segments"].items(), key=lambda kv: -kv[1]["n_users"]):
        builder.p(
            f"<b>{profile['label']}</b> — {profile['n_users']:,} users "
            f"({profile['share_of_users']*100:.1f}% of users)."
        )
        fm = profile["feature_means"]
        builder.bullets([
            f"Recency: {fm['recency_days']:.0f} days since last transaction; "
            f"active in {fm['active_months']:.1f} months on average.",
            f"{fm['n_valid_purchases']:.1f} valid purchases, avg. net amount {fm['avg_net_amount']:,.0f}.",
            f"Discount usage: {fm['discount_tx_share']*100:.1f}% of purchases; "
            f"credit card share: {fm['credit_card_share']*100:.1f}%; "
            f"loyalty member rate: {fm['loyalty_member']*100:.1f}%.",
        ])
        builder.p(f"<i>Story:</i> {_persona_story(profile, global_rates)}")
    builder.page_break()


def _add_business_questions(builder: ReportBuilder, results: dict, cfg: dict) -> None:
    alpha = cfg["analysis"]["alpha"]

    builder.h2("Q1. Repeat purchase & retention by region")
    q1 = results["q1"]
    builder.p(
        f"Chi-square test of repeat purchase vs. city: {_sig_text(q1['significant'], q1['chi2_test']['p_value'], alpha)} "
        f"(Cramer's V = {q1['chi2_test']['cramers_v']:.3f})."
    )
    if q1.get("repeat_rate_by_segment"):
        builder.p(f"By segment: {_fmt_rate_n_dict(q1['repeat_rate_by_segment'])}.")
    builder.image(q1["chart"])

    builder.h2("Q2. Repeat orders per merchant + discount impact")
    q2 = results["q2"]
    promo_split = ", ".join(f"{k}: {v:,}" for k, v in q2["promo_source_counts"].items())
    builder.p(
        f"Of {q2['n_eligible_merchants']:,} merchants with enough users to test, the platform-wide "
        f"discount rate is {q2['platform_discount_rate']*100:.1f}%. Promo-source heuristic split: "
        f"{promo_split}. {q2['note']}"
    )
    if q2.get("discount_rate_by_segment"):
        builder.p(f"Discount usage by segment: {_fmt_rate_n_dict(q2['discount_rate_by_segment'])}.")
    builder.image(q2["chart"])

    builder.h2("Q3. Best-selling vs. rarely ordered categories")
    q3 = results["q3"]
    builder.p(f"Best-selling: {q3['best_selling']}. Rarely ordered: {q3['rarely_ordered']}.")
    if q3.get("dominant_category_by_segment"):
        for seg, cats in q3["dominant_category_by_segment"].items():
            builder.p(f"{seg} top categories: {_fmt_pct_dict(cats)}.")
    builder.image(q3["chart"])

    builder.h2("Q4. Devices & minimum OS")
    q4 = results["q4"]
    builder.p(
        f"Recommended minimum Android version for {int(q4['coverage_target']*100)}% user coverage: "
        f"{q4['recommended_min_android_version']}."
    )
    if q4.get("legacy_share_by_segment"):
        builder.p(f"Legacy-OS share by segment: {_fmt_rate_n_dict(q4['legacy_share_by_segment'])}.")
    builder.image(q4["chart"])

    builder.h2("Q5. Balance vs. credit card")
    q5 = results["q5"]
    builder.p(
        f"Net amount distributions differ between payment methods: "
        f"{_sig_text(q5['significant'], q5['mann_whitney_u']['p_value'], alpha)} (Mann-Whitney U test). "
        f"Estimated total MDR cost: {q5['total_mdr_cost']:,}."
    )
    if q5.get("credit_card_share_by_segment"):
        builder.p(f"Credit-card share by segment: {_fmt_rate_n_dict(q5['credit_card_share_by_segment'])}.")
    builder.image(q5["chart"])

    builder.h2("Q6. Rating drivers")
    q6 = results["q6"]
    if q6["chi2_test"]:
        builder.p(
            f"Refund x notes association: {_sig_text(q6['chi2_test']['p_value'] < alpha, q6['chi2_test']['p_value'], alpha)} "
            f"(Cramer's V = {q6['chi2_test']['cramers_v']:.3f})."
        )
    if q6.get("rating_by_segment"):
        builder.p(f"Mean rating by segment: {_fmt_rating_n_dict(q6['rating_by_segment'])}.")
    builder.image(q6["chart"])

    builder.h2("Q7. Loyalty engagement + devices")
    q7 = results["q7"]
    builder.p(
        f"Loyalty members vs. non-members transaction frequency: "
        f"{_sig_text(q7['significant'], q7['mann_whitney_u']['p_value'], alpha)}."
    )
    if q7.get("segment_share_members"):
        builder.p(f"Segment mix among members: {_fmt_pct_dict(q7['segment_share_members'])}.")
        builder.p(f"Segment mix among non-members: {_fmt_pct_dict(q7['segment_share_non_members'])}.")
    builder.image(q7["chart"])

    builder.h2("Q8. Regions without promo usage")
    q8 = results["q8"]
    builder.p(
        f"Never-promo share varies by city: {_sig_text(q8['significant'], q8['chi2_test']['p_value'], alpha)} "
        f"(Cramer's V = {q8['chi2_test']['cramers_v']:.3f}). {q8['interpretation_note']}"
    )
    if q8.get("never_promo_share_by_segment"):
        builder.p(f"Never-promo share by segment: {_fmt_rate_n_dict(q8['never_promo_share_by_segment'])}.")
    builder.image(q8["chart"])
    builder.page_break()


def _add_feature_methodology(builder: ReportBuilder, fm: dict, feature_meta: dict) -> None:
    builder.p(
        f"{feature_meta['n_users']:,} users, {feature_meta['n_features']} engineered features per user, "
        f"built in four stages: transaction-level derivation, per-user aggregation, Big Five proxies, "
        f"and demographic proxies. Each section below states the assumption behind a feature group and "
        f"a live finding computed from this run's data, not a static description."
    )

    builder.h2("3.1 Activity (recency, frequency, regularity)")
    builder.p(
        "Assumption: how often and how recently a user transacts, and how regular that rhythm is "
        "(burstiness), reflects engagement independent of how much they spend."
    )
    a = fm["activity"]
    builder.bullets([
        f"Median active months: {a['median_active_months']:.1f} of the 3-month window.",
        f"Median tenure (first-to-last transaction): {a['median_tenure_days']:.0f} days.",
        f"Median burstiness: {a['median_burstiness']:.2f} (negative = regular spacing, positive = bursty).",
    ])

    builder.h2("3.2 Value (spend level and variability)")
    builder.p(
        "Assumption: average spend and its coefficient of variation (CV) capture both how much a user "
        "spends and how consistent that spend is, which average alone would hide."
    )
    v = fm["value"]
    builder.bullets([
        f"Median avg. net amount per purchase: {v['median_avg_net_amount']:,.0f}.",
        f"Median coefficient of variation: {v['median_cv_net_amount']:.2f}.",
    ])

    builder.h2("3.3 Category & merchant diversity")
    builder.p(
        "Assumption: Shannon entropy over a user's MCC/merchant distribution measures variety-seeking "
        "independent of how many transactions they have, unlike a raw count of distinct categories."
    )
    c = fm["category"]
    builder.bullets([
        f"Median category entropy: {c['median_category_entropy']:.2f}.",
        f"Median merchant repeat ratio: {c['median_merchant_repeat_ratio']*100:.1f}% of purchases revisit a "
        f"previously used merchant.",
    ])

    builder.h2("3.4 Geography")
    builder.p(
        "Assumption: a user's dominant transaction city approximates their home base. Finding: geo "
        "activity in this dataset is scattered rather than concentrated around one city for most users "
        "(see 3.9 for how this affected the work-location proxy)."
    )
    g = fm["geography"]
    builder.bullets([
        f"{g['multi_city_share']*100:.1f}% of users transact in more than one city.",
        f"Median share of transactions in a user's single most common city: {g['median_primary_city_share']*100:.1f}%.",
    ])

    builder.h2("3.5 Payment & promo behavior")
    builder.p("Assumption: payment-method mix and discount usage reflect financial behavior, not identity.")
    p = fm["payment"]
    builder.bullets([
        f"Balance-only users: {p['balance_only_share']*100:.1f}%; credit-card-only: {p['credit_card_only_share']*100:.1f}%.",
        f"{p['promo_user_share']*100:.1f}% of users have used a promo at least once.",
    ])

    builder.h2("3.6 Quality (failure, refund)")
    builder.p("Assumption: failure and refund rates, computed per user, surface quality issues a merchant-level view would average away.")
    q = fm["quality"]
    builder.bullets([
        f"Median per-user failure rate: {q['median_failure_rate']*100:.1f}%.",
        f"Median per-user refund rate: {q['median_refund_rate']*100:.1f}%.",
    ])

    builder.h2("3.7 Time patterns")
    builder.p("Assumption: weekend share and payday-window share capture discretionary vs. routine spending timing.")
    t = fm["time"]
    builder.bullets([
        f"Median weekend share: {t['median_weekend_share']*100:.1f}%.",
        f"Median payday-window share: {t['median_payday_share']*100:.1f}%.",
    ])

    builder.h2("3.8 Device")
    builder.p("Assumption: OS family and version are observed directly from user_agent, no inference needed.")
    d = fm["device"]
    builder.bullets([
        f"{d['android_share']*100:.1f}% of users are primarily on Android.",
        f"{d['legacy_os_share']*100:.1f}% were seen at least once on a legacy OS version.",
    ])
    builder.page_break()

    builder.h2("3.9 Big Five Personality Score")
    builder.p(
        "Methodology: mean z-score of 3-4 signed behavioral indicators per trait (table below), "
        "converted to a 0-100 percentile (src/features/personality.py)."
    )
    builder.table(
        ["Trait", "Indicators (sign)"],
        [
            ["Openness", "category_entropy (+), n_categories (+), merchant_diversity (+)"],
            ["Conscientiousness", "burstiness (-), failure_rate (-), essential_share (+), payday_share (-)"],
            ["Extraversion", "social_share (+), weekend_share (+), n_cities (+)"],
            ["Agreeableness", "avg_rating (+), merchant_repeat_ratio (+), notes_share (-)"],
            ["Neuroticism", "refund_rate (+), rating_std (+), cv_net_amount (+)"],
        ],
    )
    builder.p(
        "Assumption: this indicator-to-trait mapping is this project's own hypothesis (ordinary "
        "consumer-behavior reasoning, e.g. shopping across more categories suggests openness), not "
        "copied from a validated instrument. It is grounded in a real research precedent for the "
        "general idea, not a specific cited method: Gladstone, Matz & Lemaire (2019, Psychological "
        "Science) linked real UK bank transactions to a self-reported Big Five questionnaire (BFI-10) "
        "and found Extraversion most positively correlated with dining/drinking spend, which matches "
        "this project's social_share indicator for the same trait; the other traits here use "
        "categories that study's data had and this dataset's 21 MCC codes do not, so there was nothing "
        "to compare for those. Ramon, Matz, Farrokhnia & Martens (2021) found Conscientiousness and "
        "Neuroticism most predictable from spending data and Openness/Agreeableness least predictable; "
        "Neuroticism is one of two traits that pass the reliability check below, which is at least "
        "directionally consistent."
    )
    rows = [
        [trait, m["cronbach_alpha"], m["mean_inter_indicator_correlation"], "Yes" if m["reliable"] else "No"]
        for trait, m in feature_meta["personality"].items()
    ]
    builder.table(["Trait", "Cronbach's alpha", "Mean inter-item corr.", "Reliable (alpha >= 0.5)"], rows)
    builder.p(
        "Finding: Cronbach's alpha only checks whether a trait's own indicators move together in this "
        "dataset; it cannot confirm the score matches a real person's personality, since no real "
        "personality label exists here to check against. A trait with alpha below 0.5 should not be "
        "interpreted as a real personality measurement; its indicators do not move together enough to "
        "justify combining them. Both studies cited above validated against a real self-report "
        "questionnaire the account holder filled in, not a third-party rater's impression; that is the "
        "validation path this project would need (e.g. a short BFI-10 or TIPI survey at onboarding), "
        "not crowd annotation, since transaction history gives an outside observer nothing to watch the "
        "way a video would."
    )
    builder.page_break()

    builder.h2("3.10 Demographic Proxies")
    builder.p(
        "Every demographic attribute below gets at least one data-driven proxy attempt, with the "
        "assumption behind it stated explicitly, except Industry of Employment, which is documented "
        "as infeasible after an explicit attempt (no field ties a user to an employer at all). Every "
        "proxy ships a confidence column so a weak signal is visible rather than hidden."
    )
    for item in fm["attribute_table"]:
        builder.p(f"<b>{item['attribute']}</b> ({item['priority']} priority) — columns: {item['columns']}")
        builder.bullets([
            f"Methodology: {item['methodology']}",
            f"Assumption: {item['assumption']}",
            f"Finding: {item['finding']}",
        ])
    builder.page_break()


def _add_assumptions(builder: ReportBuilder, cfg: dict) -> None:
    builder.bullets([
        "transaction_amount is assumed to be the gross amount (before discount); net_amount = "
        "transaction_amount - promo_amount.",
        f"MDR cost assumptions: balance = {cfg['assumptions']['mdr']['balance']*100:.2f}%, "
        f"credit_card = {cfg['assumptions']['mdr']['credit_card']*100:.2f}% of transaction value.",
        f"Eid al-Fitr period: {cfg['calendar']['lebaran_start']} to {cfg['calendar']['lebaran_end']}.",
        "Every requested demographic attribute except Industry of Employment has a heuristic proxy "
        "(Chapter 3.10); confidence is mostly 'low' or 'none' by design because the underlying signals "
        "are weak. None of them should be treated as a validated classification — see "
        "demographic_feasibility.csv for the one attribute left undone and why.",
        "For real-world use, the biggest gaps are: a genuine promo-source field, a resolved status for "
        "the 33.6% of transactions currently blank, and self-reported ground truth (e.g. a short BFI-10 "
        "or TIPI survey) to actually validate the Big Five and demographic proxies against real labels.",
    ])


def run_report(cfg) -> dict:
    logger.info("Loading enriched transactions, user features, segments, and DQ report")
    transactions = read_table(cfg.path("transactions_enriched"))
    user_features = read_table(cfg.path("user_features"))
    user_segments = read_table(cfg.path("user_segments"))
    segment_meta = read_json(cfg.path("segment_profiles"))
    dq_report = read_json(cfg.path("dq_report"))
    feature_meta = read_json(cfg.path("feature_meta"))

    user_features = user_features.merge(user_segments, on="user_id", how="left")
    transactions = transactions.merge(user_segments, on="user_id", how="left")

    figures_dir = cfg.path("figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running analyses for the 8 business questions")
    results = run_all_analyses(transactions, user_features, dq_report, segment_meta, cfg.raw, figures_dir)

    write_json(results, cfg.path("analysis_results"))
    logger.info("Wrote analysis results to %s", cfg.path("analysis_results"))

    logger.info("Building PDF report")
    _build_pdf(results, feature_meta, cfg, figures_dir)
    logger.info("Wrote PDF report to %s", cfg.path("report_pdf"))

    return results
