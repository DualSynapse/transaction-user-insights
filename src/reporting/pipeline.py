"""run_report(cfg): analyses -> charts -> PDF."""
import logging

from src.reporting.analyses import run_all_analyses
from src.reporting.pdf_builder import ReportBuilder
from src.utils.io import read_json, read_table, write_json

logger = logging.getLogger("pipeline")


def _sig_text(significant: bool, p_value: float, alpha: float) -> str:
    if significant:
        return f"statistically significant (p = {p_value:.4f} < alpha = {alpha})"
    return f"not statistically significant (p = {p_value:.4f} >= alpha = {alpha})"


def _executive_summary(results: dict, cfg: dict) -> list[str]:
    audit = results["audit"]
    q3 = results["q3"]
    q4 = results["q4"]
    q7 = results["q7"]
    findings = [
        f"{audit['n_transactions']:,} transactions from {audit['n_users']:,} users across "
        f"{audit['n_merchants']:,} merchants were processed, covering {audit['date_range'][0][:10]} "
        f"to {audit['date_range'][1][:10]}.",
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
        "statistically significant. Chapter 7 (assumptions & limitations) covers this in detail.",
    ]
    return findings


def _build_pdf(results: dict, feature_meta: dict, cfg, figures_dir) -> None:
    audit = results["audit"]
    period = f"{audit['date_range'][0][:10]} to {audit['date_range'][1][:10]}"
    builder = ReportBuilder(cfg.path("report_pdf"), "Transaction User Insights Report", period)

    builder.h1("1. Executive Summary")
    builder.bullets(_executive_summary(results, cfg.raw))
    builder.page_break()

    builder.h1("2. Data Audit")
    builder.p(
        f"The dataset contains {audit['n_transactions']:,} transactions, {audit['n_users']:,} users, "
        f"{audit['n_merchants']:,} merchants and {audit['n_mcc_categories']} MCC categories."
    )
    builder.h2("Transaction status breakdown")
    builder.table(["Status", "Count"], [[k, f"{v:,}"] for k, v in audit["status_breakdown"].items()])
    builder.h2("Data quality flags")
    builder.table(
        ["Flag", "Count"],
        [[k, f"{v:,}"] for k, v in audit["dq_flag_counts"].items() if v > 0],
    )
    builder.h2("Limitations")
    builder.bullets(audit["limitations"])
    builder.page_break()

    builder.h1("3. Regional Profile")
    builder.p(f"Geo-matching coverage: {audit['geo_match_coverage_pct']}% of transactions matched to a regency/city.")
    builder.image(results["region_map_chart"])
    builder.page_break()

    builder.h1("4. Time Patterns")
    builder.image(results["daily_volume_chart"])
    builder.p(
        "Volume rises markedly during the Eid al-Fitr period (7-13 April 2024), consistent with "
        "the initial data exploration."
    )
    builder.page_break()

    builder.h1("5. Business Questions")
    _add_business_questions(builder, results, cfg.raw)

    builder.h1("6. User Feature Summary")
    _add_feature_summary(builder, feature_meta)

    builder.h1("7. Assumptions & Limitations")
    _add_assumptions(builder, cfg.raw)

    builder.build()


def _add_business_questions(builder: ReportBuilder, results: dict, cfg: dict) -> None:
    alpha = cfg["analysis"]["alpha"]

    builder.h2("Q1. Repeat purchase & retention by region")
    q1 = results["q1"]
    builder.p(
        f"Chi-square test of repeat purchase vs. city: {_sig_text(q1['significant'], q1['chi2_test']['p_value'], alpha)} "
        f"(Cramer's V = {q1['chi2_test']['cramers_v']:.3f})."
    )
    builder.image(q1["chart"])

    builder.h2("Q2. Repeat orders per merchant + discount impact")
    q2 = results["q2"]
    builder.p(
        f"Of {q2['n_eligible_merchants']} merchants with enough users to test, the platform-wide "
        f"discount rate is {q2['platform_discount_rate']*100:.1f}%. Promo-source heuristic split: "
        f"{q2['promo_source_counts']}. {q2['note']}"
    )
    builder.image(q2["chart"])

    builder.h2("Q3. Best-selling vs. rarely ordered categories")
    q3 = results["q3"]
    builder.p(f"Best-selling: {q3['best_selling']}. Rarely ordered: {q3['rarely_ordered']}.")
    builder.image(q3["chart"])

    builder.h2("Q4. Devices & minimum OS")
    q4 = results["q4"]
    builder.p(
        f"Recommended minimum Android version for {int(q4['coverage_target']*100)}% user coverage: "
        f"{q4['recommended_min_android_version']}."
    )
    builder.image(q4["chart"])

    builder.h2("Q5. Balance vs. credit card")
    q5 = results["q5"]
    builder.p(
        f"Net amount distributions differ between payment methods: "
        f"{_sig_text(q5['significant'], q5['mann_whitney_u']['p_value'], alpha)} (Mann-Whitney U test). "
        f"Estimated total MDR cost: {q5['total_mdr_cost']:,}."
    )
    builder.image(q5["chart"])

    builder.h2("Q6. Rating drivers")
    q6 = results["q6"]
    if q6["chi2_test"]:
        builder.p(
            f"Refund x notes association: {_sig_text(q6['chi2_test']['p_value'] < alpha, q6['chi2_test']['p_value'], alpha)} "
            f"(Cramer's V = {q6['chi2_test']['cramers_v']:.3f})."
        )
    builder.image(q6["chart"])

    builder.h2("Q7. Loyalty engagement + devices")
    q7 = results["q7"]
    builder.p(
        f"Loyalty members vs. non-members transaction frequency: "
        f"{_sig_text(q7['significant'], q7['mann_whitney_u']['p_value'], alpha)}."
    )
    builder.image(q7["chart"])

    builder.h2("Q8. Regions without promo usage")
    q8 = results["q8"]
    builder.p(
        f"Never-promo share varies by city: {_sig_text(q8['significant'], q8['chi2_test']['p_value'], alpha)} "
        f"(Cramer's V = {q8['chi2_test']['cramers_v']:.3f}). {q8['interpretation_note']}"
    )
    builder.image(q8["chart"])
    builder.page_break()


def _add_feature_summary(builder: ReportBuilder, feature_meta: dict) -> None:
    builder.p(f"{feature_meta['n_users']:,} users, {feature_meta['n_features']} engineered features per user.")
    builder.h2("Big Five proxy reliability (Cronbach's alpha)")
    rows = [
        [trait, m["cronbach_alpha"], m["mean_inter_indicator_correlation"], "Yes" if m["reliable"] else "No"]
        for trait, m in feature_meta["personality"].items()
    ]
    builder.table(["Trait", "Cronbach's alpha", "Mean inter-item corr.", "Reliable (alpha >= 0.5)"], rows)
    builder.p(
        "A trait with alpha below 0.5 should not be interpreted as a real personality measurement; "
        "its indicators do not move together enough to justify combining them."
    )
    builder.page_break()


def _add_assumptions(builder: ReportBuilder, cfg: dict) -> None:
    builder.bullets([
        "transaction_amount is assumed to be the gross amount (before discount); net_amount = "
        "transaction_amount - promo_amount.",
        f"MDR cost assumptions: balance = {cfg['assumptions']['mdr']['balance']*100:.2f}%, "
        f"credit_card = {cfg['assumptions']['mdr']['credit_card']*100:.2f}% of transaction value.",
        f"Eid al-Fitr period: {cfg['calendar']['lebaran_start']} to {cfg['calendar']['lebaran_end']}.",
        "Demographic attributes without a supporting signal (gender, age, education, marital status, "
        "work location, working status, industry) are not estimated; see demographic_feasibility.csv "
        "for what data would be required.",
        "For real-world use, the biggest gaps are: a genuine promo-source field, a resolved status for "
        "the 33.6% of transactions currently blank, and a validated demographic data source if "
        "personalization beyond behavior-based signals is required.",
    ])


def run_report(cfg) -> dict:
    logger.info("Loading enriched transactions, user features, and DQ report")
    transactions = read_table(cfg.path("transactions_enriched"))
    user_features = read_table(cfg.path("user_features"))
    dq_report = read_json(cfg.path("dq_report"))
    feature_meta = read_json(cfg.path("feature_meta"))

    figures_dir = cfg.path("figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running analyses for the 8 business questions")
    results = run_all_analyses(transactions, user_features, dq_report, cfg.raw, figures_dir)

    write_json(results, cfg.path("analysis_results"))
    logger.info("Wrote analysis results to %s", cfg.path("analysis_results"))

    logger.info("Building PDF report")
    _build_pdf(results, feature_meta, cfg, figures_dir)
    logger.info("Wrote PDF report to %s", cfg.path("report_pdf"))

    return results
