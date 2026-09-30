# Transaction User Insights

Turns the raw transaction dataset into clean, geo-enriched transaction data, a per-user
feature table (including Big Five proxies and demographic signals), and an automated PDF
report covering a data audit and eight business questions.

## Setup

```bash
pip install -r requirements.txt
```

geopandas requires GDAL/GEOS/PROJ; if it fails to install, the pipeline still runs using a
centroid-based fallback for geo enrichment (see `src/cleaning/geo_enrichment.py`).

## Usage

```bash
python main.py                      # run all stages
python main.py --stage clean        # cleaning + geo enrichment only
python main.py --stage features     # feature engineering only (requires clean output)
python main.py --stage report       # report only (requires features output)
python main.py --config other.yaml  # use a different config file
```

Each stage reads the previous stage's output from disk, so stages can be re-run
independently. If a required input is missing, the pipeline exits with a clear message
telling you which stage to run first.

## Pipeline

```
data/raw/transactions.csv
      |
      v  [clean]     src/cleaning/pipeline.py
data/interim/transactions_clean.parquet
outputs/reports/data_quality_report.json
      |
      v  [features]  src/features/pipeline.py
data/processed/user_features.parquet
data/processed/feature_dictionary.csv
data/processed/demographic_feasibility.csv
data/processed/feature_meta.json
      |
      v  [report]    src/reporting/pipeline.py
outputs/reports/transaction_insights_report.pdf
outputs/reports/analysis_results.json
outputs/figures/*.png
```

## Configuration

All parameters (paths, missing-value tokens, category values, geo settings, calendar
windows, MDR assumptions, significance level) live in `config.yaml` — no magic numbers in
the code.

## Cleaning principle

Flag, don't delete. Original columns are never overwritten; data quality issues are
recorded in `dq_*` boolean columns (combined into `dq_any_issue`), and non-error notes are
recorded in `info_*` columns.

## Geo enrichment

Uses geopandas to match each transaction's coordinates to a regency/city polygon from the
GADM boundary file (`rifani/geojson-political-indonesia`, cached in `data/external/` after
the first run). Matching order: `within` (point-in-polygon) -> `nearest` (within
`geo.nearest_max_km`) -> `centroid_fallback` (if geopandas/the boundary file is
unavailable) -> `unmatched`. The method used is recorded per transaction in
`geo_match_method`.

## Testing

```bash
pytest
```

Tests use small synthetic fixtures and don't need the original dataset (the end-to-end
smoke test generates its own sample CSV, but reuses the already-cached boundary file in
`data/external/` to avoid a network call).

## Known limitations

- The dataset is synthetic: 33.6% of transactions have a blank status (treated as a third
  "pending" category rather than as missing data), and most cross-variable relationships
  tested in the report are weak or not statistically significant.
- There is no promo-source column, so merchant- vs. platform-funded discounts (business
  question 2) are inferred heuristically.
- Demographic attributes without a supporting signal in the data (gender, age, education,
  marital status, work location, working status, industry) are not estimated; see
  `data/processed/demographic_feasibility.csv` for what data would be required.
- User segmentation is not yet part of the pipeline; it is a planned follow-up phase once
  feature engineering is stable (see `IMPLEMENTATION_PLAN (1).md`, phase 7).
