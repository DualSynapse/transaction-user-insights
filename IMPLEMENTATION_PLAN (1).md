# Implementation Plan: Transaction User Insights

## 1. Objective

Build a single repository that turns the raw transaction dataset into:

1. **Clean transaction data** enriched with province and regency/city.
2. **A per-user feature table**, including Big Five proxies and demographic signals.
3. **An automated PDF report** containing a data audit and answers to the business questions.

The entire process runs with a single command: `python main.py`.

## 2. Agreed decisions

| Decision | Choice |
|---|---|
| Report format | PDF |
| User segmentation | Not yet part of the pipeline; added as a follow-up phase once feature engineering is complete |
| Geo enrichment | Uses geopandas in the cleaning stage, adding province and regency/city columns |
| Cleaning principle | Flag, don't delete. Original columns are never overwritten |

## 3. Data context (from initial exploration)

- 278,932 transactions, 24,321 users, 6,489 merchants, 21 MCC categories, covering 1 March to 31 May 2024.
- **No missing values** (including hidden tokens such as `""`, `null`, `N/A`). `promo_amount = 0` and `transaction_notes = no` are valid values, not missing data.
- `transaction_date` contains the date only (no time). `transaction_notes` contains only yes/no (no free text).
- Inconsistencies found: ratings on failed transactions (14,025), refunds on failed transactions (708), discount "yes" with zero promo amount (218).
- The only strong signal: volume rises roughly 2.5x on 7-13 April 2024 (Eid al-Fitr 1445 H).
- A geopandas test has already been run: **100% of transactions were mapped** to 9 regencies/cities across 6 provinces. The points previously labelled "Yogyakarta" actually fall within **Bantul Regency**.

## 4. Repository structure

```
transaction-user-insights/
├── main.py                      # Entry point for the whole pipeline
├── config.yaml                  # All parameters and paths
├── requirements.txt
├── README.md
├── IMPLEMENTATION_PLAN.md
│
├── data/
│   ├── raw/                     # transactions.csv (input, never modified)
│   ├── external/                # Administrative boundary GeoJSON (auto-download + cache)
│   ├── interim/                 # transactions_clean.parquet
│   └── processed/               # user_features.parquet, feature_dictionary.csv
│
├── src/
│   ├── utils/
│   │   ├── config.py            # Load config.yaml, resolve absolute paths
│   │   ├── logger.py            # Logging to console + file
│   │   └── io.py                # Read/write Parquet (CSV.gz fallback), JSON
│   │
│   ├── cleaning/
│   │   ├── audit.py             # Missing-value and whitespace audit
│   │   ├── standardize.py       # Data types, category validation, booleans
│   │   ├── consistency.py       # Duplicates, value ranges, dq_* flags
│   │   ├── geo_enrichment.py    # geopandas: province, regency/city, region
│   │   ├── device_parser.py     # user_agent -> OS, device, version
│   │   └── pipeline.py          # run_cleaning(cfg)
│   │
│   ├── features/
│   │   ├── transaction_level.py # MCC group, net amount, time flags
│   │   ├── user_level.py        # Per-user aggregation
│   │   ├── personality.py       # Big Five proxies + Cronbach's alpha
│   │   ├── demographics.py      # Demographic signals + feasibility table
│   │   └── pipeline.py          # run_features(cfg)
│   │
│   └── reporting/
│       ├── analyses.py          # Data audit + 8 business questions + statistical tests
│       ├── charts.py            # matplotlib charts (PNG)
│       ├── pdf_builder.py       # PDF components (reportlab): cover, headings, tables, images
│       └── pipeline.py          # run_report(cfg)
│
├── outputs/
│   ├── reports/                 # transaction_insights_report.pdf, data_quality_report.json
│   ├── figures/                 # All PNG charts
│   └── logs/                    # pipeline.log
│
└── tests/
    ├── test_cleaning.py
    ├── test_geo_enrichment.py
    ├── test_features.py
    └── test_pipeline_smoke.py
```

## 5. Pipeline flow and `main.py`

```
transactions.csv
      │
      ▼  [clean]     src/cleaning/pipeline.py
transactions_clean.parquet + data_quality_report.json
      │
      ▼  [features]  src/features/pipeline.py
user_features.parquet + feature_dictionary.csv + feature_meta.json
      │
      ▼  [report]    src/reporting/pipeline.py
transaction_insights_report.pdf + figures/*.png
```

Each stage reads the previous stage's output from disk, so stages can be re-run independently:

```bash
python main.py                      # all stages
python main.py --stage clean
python main.py --stage features
python main.py --stage report
python main.py --config other.yaml
```

`main.py` behaviour:
- Loads `config.yaml` and sets up logging to `outputs/logs/pipeline.log`.
- Runs the stages in order and logs the duration of each.
- If a previous stage's output is missing, shows a clear message (e.g. "run the clean stage first") and exits with code 1.
- An error in any stage is logged in full (traceback), and the pipeline stops.

## 6. Configuration (`config.yaml`)

| Section | Contents |
|---|---|
| `paths` | Locations of input, intermediate outputs, final outputs, and logs |
| `cleaning` | Missing-value tokens, allowed category values, legacy OS version thresholds |
| `geo` | Boundary file and URL, polygon column names, maximum fallback distance, name normalisation, regional grouping, fallback centroids |
| `calendar` | Eid al-Fitr period, payday window |
| `assumptions` | Definition of `transaction_amount` (before discount), assumed MDR costs for payment simulation |
| `analysis` | Significance level, minimum users per merchant, OS coverage target |

No magic numbers in the code; every parameter that may change lives in the config.

## 7. Stage 1: Cleaning

### 7.1 Steps

| # | Module | Process | Output |
|---|---|---|---|
| 1 | `audit.py` | Read the CSV as strings; count NaN + hidden missing tokens + extra whitespace per column | `missing_audit` in the DQ report |
| 2 | `standardize.py` | Trim whitespace, missing tokens -> NaN, validate category values, convert numerics and dates, yes/no -> boolean | `*_flag` columns, `is_completed` |
| 3 | `consistency.py` | Duplicate `transaction_id`, range checks (amount > 0, promo <= amount, rating 1-5), cross-column consistency checks | `dq_*` columns, `merchant_rating_clean` |
| 4 | `geo_enrichment.py` | Parse coordinates, point-in-polygon to regency/city | Geo columns (see 7.2) |
| 5 | `device_parser.py` | Parse `user_agent` (unique values only, for speed) | `os_family`, `device_type`, `os_major`, `os_version`, `info_legacy_os` |
| 6 | `pipeline.py` | Combine flags, save data and report | `transactions_clean.parquet`, `data_quality_report.json` |

Flag rules:
- `dq_*` = data errors or inconsistencies; combined into `dq_any_issue`.
- `info_*` = notes that are not errors (e.g. legacy OS version, non-standard MCC); not counted as data issues.

### 7.2 Geo enrichment with geopandas

**Boundary source:** GADM regency/city GeoJSON from the GitHub repository `rifani/geojson-political-indonesia`. The file is downloaded once and cached in `data/external/`, so subsequent runs need no internet access. The source can be swapped via the config.

**Matching order** (recorded in the `geo_match_method` column):

| Order | Method | When it applies |
|---|---|---|
| 1 | `within` | The point lies inside a regency/city polygon (`gpd.sjoin`, `predicate="within"`) |
| 2 | `nearest` | The point is outside all polygons (sea, border); take the nearest polygon within `nearest_max_km` (`gpd.sjoin_nearest`) |
| 3 | `centroid_fallback` | The boundary file is unavailable or geopandas is not installed; use city centroids from the config |
| 4 | `unmatched` | No method matched; flagged with `dq_geo_unmatched` |

Matching runs on **unique coordinates**, then the result is merged back into the transaction table.

**Name normalisation:** "Jakarta Raya" -> "DKI Jakarta", "Yogyakarta" -> "DI Yogyakarta", "Kotamadya" -> "Kota", and regency/city names are prefixed with "Kota"/"Kabupaten" where missing.

**New columns:**

| Column | Example |
|---|---|
| `lat`, `lon` | -6.4021, 106.7943 |
| `province` | Jawa Barat |
| `city` | Kota Depok |
| `city_type` | Kota (city) / Kabupaten (regency) |
| `region_group` | Jabodetabek / Central Java & DIY / East Java |
| `geo_match_method` | within / nearest / centroid_fallback / unmatched |
| `dq_geo_unparseable`, `dq_geo_outside_indonesia`, `dq_geo_unmatched` | Geo quality flags |

**Initial test results:**

| Province | Regency/City | Transactions |
|---|---|---|
| DKI Jakarta | Kota Jakarta Pusat | 72,992 |
| Jawa Timur | Kota Surabaya | 38,896 |
| Banten | Kota Tangerang | 28,826 |
| Jawa Barat | Kota Depok | 27,289 |
| Jawa Barat | Kota Bekasi | 27,244 |
| Jawa Barat | Kota Bogor | 25,697 |
| DI Yogyakarta | Kabupaten Bantul | 24,518 |
| Jawa Timur | Kota Malang | 16,877 |
| Jawa Tengah | Kota Semarang | 16,592 |
| DKI Jakarta | Kota Jakarta Selatan | 1 |

Note: this GADM file contains 394 regencies/cities (predating several administrative splits; there are 514 today). It has no effect on the 9 cities in this dataset, but it should be mentioned in the report.

## 8. Stage 2: Feature engineering

### 8.1 Transaction-level derived columns (`transaction_level.py`)

`mcc_description`, `mcc_group`, `info_mcc_non_standard` (MCC 5412), `net_amount`, `discount_rate`, `is_valid_purchase` (completed and not refunded), `is_discount_clean`, `year_month`, `week`, `day_of_week`, `is_weekend`, `is_payday_window`, `is_lebaran_period`.

### 8.2 Per-user features (`user_level.py`)

| Group | Features |
|---|---|
| Activity | `n_transactions`, `n_valid_purchases`, `active_days`, `active_months`, `tenure_days`, `recency_days`, `avg_gap_days`, `burstiness` |
| Value | `total_net_amount`, `avg_net_amount`, `median_net_amount`, `max_net_amount`, `cv_net_amount` |
| Category | `n_categories`, `category_entropy`, `n_merchants`, `merchant_diversity`, `merchant_repeat_ratio`, `share_<mcc_group>` |
| Geography | `primary_city`, `primary_city_share`, `primary_province`, `primary_region_group`, `n_cities`, `n_provinces`, `is_multi_city`, `is_multi_province` |
| Payment & promo | `credit_card_share`, `payment_profile`, `discount_tx_share`, `total_promo_amount`, `avg_discount_rate_when_used`, `is_promo_user`, `loyalty_member` |
| Quality | `failure_rate`, `refund_rate`, `avg_rating`, `rating_std`, `notes_share` |
| Time | `weekend_share`, `payday_share`, `lebaran_share`, `lebaran_lift` |
| Device | `os_family`, `device_type`, `os_version`, `is_legacy_os` |

### 8.3 Big Five proxies (`personality.py`)

Each trait = mean z-score of its indicators (with +/- direction), converted to a 0-100 percentile. Each indicator is used in only one trait.

| Trait | Indicators |
|---|---|
| Openness | + category_entropy, + n_categories, + merchant_diversity |
| Conscientiousness | - burstiness, - failure_rate, + essential_share, - payday_share |
| Extraversion | + social_share, + weekend_share, + n_cities |
| Agreeableness | + avg_rating, + merchant_repeat_ratio, - notes_share |
| Neuroticism | + refund_rate, + rating_std, + cv_net_amount |

Validation: **Cronbach's alpha** and mean inter-indicator correlation per trait, stored in `feature_meta.json` and shown in the report. An alpha below 0.5 means the trait score should not be interpreted as personality.

### 8.4 Demographic signals (`demographics.py`)

Columns are created only for attributes that have supporting signals in the dataset:

| Column | Supporting MCC |
|---|---|
| `parental_signal` | 8211 (schools) |
| `student_signal` | 8220 (colleges/universities) |
| `vehicle_signal` | 5541 (fuel stations) |
| `homeowner_signal` | 5251, 5712 (hardware, furniture) |
| `health_care_signal` | 5912 (pharmacies) |
| `active_lifestyle_signal` | 5941 (sporting goods) |
| `home_city_estimate` | Dominant city + confidence from its share |
| `income_proxy_tier` | Credit card share + amount percentile (weak) |

Each signal has `*_count` and `*_confidence` (none/low/medium/high) columns and is compared against a **chance baseline** (the probability a user would be flagged if categories were picked according to their global frequency). If the observed rate is close to the baseline, the signal is no better than chance.

Infeasible attributes (gender, age, education, marital status, work location, working status, industry) **do not get columns**, but are documented in `demographic_feasibility.csv` with the reason and the data that would be needed.

### 8.5 Outputs

`user_features.parquet`, `feature_dictionary.csv` (name, group, description, type, missing share), `demographic_feasibility.csv`, `feature_meta.json`, and `transactions_enriched.parquet` (used by the report stage).

## 9. Stage 3: Automated PDF report

### 9.1 Technology

- **reportlab** (Platypus) to assemble the PDF: cover, table of contents, headings, paragraphs, tables, images.
- **matplotlib** for charts, saved as PNG in `outputs/figures/` and embedded in the PDF.
- **scipy** for statistical tests (chi-square, Kruskal-Wallis, Mann-Whitney) and effect sizes (Cramer's V).
- Built-in Helvetica font; avoid characters outside Latin-1 (e.g. arrows, the >= sign) so they don't render as black boxes.
- Finding text is generated dynamically from the analysis results, with different wording for significant vs non-significant results.

### 9.2 Report structure

| Chapter | Contents |
|---|---|
| Cover | Title, data period, generation date |
| 1. Executive summary | 4-6 key findings |
| 2. Data audit | Data profile, `dq_*` flags, unrealistic columns, limitations |
| 3. Regional profile | Map of transaction points per regency/city, province/city table |
| 4. Time patterns | Daily volume, Eid al-Fitr effect |
| 5. Business questions 1-8 | Each section: metric, chart, statistical test, conclusion, recommendation |
| 6. User feature summary | Distribution of key features, Big Five Cronbach's alpha, demographic feasibility table |
| 7. Assumptions & limitations | Definition assumptions, additional data needed for real-world use |

### 9.3 Business question mapping

| # | Question | Metrics & method |
|---|---|---|
| 1 | Repeat purchase & retention by region | Repeat rate per transaction city; March cohort retention per user's dominant city; chi-square |
| 2 | Repeat orders per merchant + discount impact | Repeat-user rate per merchant (minimum users set in config); merchant vs platform promo heuristic (binomial test of each merchant's discount rate against the baseline); note that a promo-source column is not available |
| 3 | Best-selling vs rarely ordered categories | Transaction count, GMV, unique users, March-to-May growth, Eid al-Fitr lift |
| 4 | Devices & minimum OS | Users and GMV per OS/device, OS version coverage curve, recommended minimum version at the coverage target, failure rate per OS |
| 5 | Balance vs credit card | Share, AOV, refund/failure/discount per method, user profile (balance only / credit card only / mixer), cost simulation using MDR assumptions |
| 6 | Rating drivers | Mean clean rating per factor, chi-square + Cramer's V, refund x notes table |
| 7 | Loyalty engagement + devices | Members vs non-members (frequency, value, retention), devices of the most engaged users |
| 8 | Regions without promo usage | Share of users who never used a promo per city, chi-square, interpretation (availability vs price sensitivity vs awareness) |

## 10. Testing

Uses **pytest** with a small synthetic dataset (fixtures), so tests run fast and don't need the original file:

| File | What is tested |
|---|---|
| `test_cleaning.py` | Hidden missing tokens are detected, consistency flags are correct, user_agent parser |
| `test_geo_enrichment.py` | Points inside a synthetic polygon, nearest fallback, centroid fallback without a boundary file, name normalisation |
| `test_features.py` | Burstiness, entropy, category shares, Cronbach's alpha, signal confidence |
| `test_pipeline_smoke.py` | `main.py` runs end-to-end on sample data and produces a PDF |

## 11. Dependencies (`requirements.txt`)

```
pandas>=2.0
numpy>=1.24
pyarrow>=14
pyyaml>=6
geopandas>=0.14
shapely>=2.0
scipy>=1.10
matplotlib>=3.7
reportlab>=4.0
pytest>=7
```

## 12. Delivery phases

| Phase | Work | Definition of done |
|---|---|---|
| 1 | Repo skeleton, `config.yaml`, utils, `main.py` | `python main.py --help` runs; log file is created |
| 2 | Cleaning modules + geo enrichment | `transactions_clean.parquet` has province and regency/city columns; `geo_match_method` is 100% populated; DQ report matches the exploration figures |
| 3 | Feature engineering modules | One row per user (24,321); `feature_dictionary.csv` complete; Cronbach's alpha recorded |
| 4 | Analysis of the 8 business questions + charts | `analysis_results.json` and PNGs created for every question |
| 5 | PDF builder + report pipeline | Complete PDF with all chapters; figures in the PDF match the JSON |
| 6 | Testing, README, end-to-end run | All tests pass; `python main.py` from a clean repo produces all outputs |
| 7 (follow-up) | Segmentation module (`src/segmentation/`) as a new stage before the report | To be defined after phase 3 is complete |

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No internet access when downloading boundaries | File is cached and included in the repo; automatic centroid fallback |
| geopandas is hard to install in some environments | Import wrapped in try/except; the pipeline still runs with the centroid fallback |
| pyarrow is unavailable | Automatically saves as CSV.gz |
| The data is synthetic, so there are almost no relationships between variables | The report shows statistical tests and effect sizes as they are; each chapter separates "findings" from "methods ready for real data" |
| The definition of `transaction_amount` (before/after discount) is uncertain | Made a config parameter and listed as an assumption in the report |
