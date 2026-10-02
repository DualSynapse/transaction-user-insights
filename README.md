# Transaction User Insights

Turns the raw transaction dataset into clean, geo-enriched transaction data, a per-user
feature table (including Big Five proxies and 14 demographic proxies), a behavioral user
segmentation (K-Means), and an automated PDF report.

## Submission layout

The code (this repository: `src/`, `notebooks/`, `main.py`, `tests/`) and the report are
kept as two separate deliverables:

1. **Code**: everything in this repository is the exploration, feature-creation, and
   segmentation code. `notebooks/` is the exploratory side; `src/` is the production
   pipeline the notebooks call into (see "Notebooks" below for how the two relate).
2. **Report**: `outputs/reports/PROJECT_REPORT.pdf`, a first-person narrative document
   covering the reasoning and methodology behind every engineered feature (including
   assumptions and findings) and a summary of the user segments with each one's key
   characteristics. `outputs/reports/transaction_insights_report.pdf` is a second,
   auto-generated PDF built by the pipeline itself (see below); Chapters 3 and 4 of that
   PDF cover the same feature-methodology and segmentation ground in a more data-table-heavy
   form, with everything beyond that scope (data audit, regional/time patterns, 8 business
   questions) kept in a clearly labeled supplementary appendix (Chapter 6).

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended - installs into a local `.venv` and
pins exact versions via `uv.lock`):

```bash
uv sync
```

Or with plain pip:

```bash
pip install -r requirements.txt
```

geopandas requires GDAL/GEOS/PROJ; if it fails to install, the pipeline still runs using a
centroid-based fallback for geo enrichment (see `src/cleaning/geo_enrichment.py`).

## Usage

```bash
uv run python main.py                      # run all stages
uv run python main.py --stage clean        # cleaning + geo enrichment only
uv run python main.py --stage features     # feature engineering only (requires clean output)
uv run python main.py --stage segment      # user segmentation only (requires features output)
uv run python main.py --stage report       # report only (requires segment output)
uv run python main.py --config other.yaml  # use a different config file
```

(Drop the `uv run` prefix if you installed with pip into an already-active environment.)

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
      v  [segment]   src/segmentation/pipeline.py
data/processed/user_segments.parquet
data/processed/segment_profiles.json
data/processed/segmentation_model.joblib
      |
      v  [report]    src/reporting/pipeline.py
outputs/reports/transaction_insights_report.pdf
outputs/reports/analysis_results.json
outputs/figures/*.png
```

## Repository layout: tracked vs. regenerated

`data/raw/` (the input), `data/external/` (the cached geo boundary file), `notebooks/`,
`src/`, and `outputs/reports/` and `outputs/figures/` (the final PDF, charts, and JSON
results) are tracked in git. `data/interim/` and `data/processed/` are not: every file in
them is fully reproducible by running the pipeline, so they stay as regenerated build
artifacts rather than committed files. `outputs/logs/` is also excluded, since a log file
is only meaningful for the run that produced it.

## Configuration

All parameters (paths, missing-value tokens, category values, geo settings, calendar
windows, MDR assumptions, significance level) live in `config.yaml` - no magic numbers in
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

## User segmentation

`src/segmentation/` clusters users with K-Means on **behavioral features only**
(recency, activity, value, category/merchant diversity, discount & loyalty usage,
payment mix, quality) - geography, device, payment-method category, and the Big Five
proxy scores are excluded from the clustering inputs and used only to *profile* the
resulting segments, so a segment reflects behavior rather than a relabeled existing
column. The number of clusters (k) is chosen automatically in the `k_min`-`k_max` range
from `config.yaml` by silhouette score. Segments are named by ranking cluster centroids
on a composite recency/frequency/value score (e.g. "Champions (High-Value Loyal)",
"At-Risk / Dormant", "Bargain Hunters", "New / Low Engagement", "Regular / Steady
Spenders"). `notebooks/03_user_segmentation.ipynb` documents the feature-selection
rationale and k-selection experiments; each segment's report entry (Chapter 4) includes a
persona story built from which rare demographic proxies it over-indexes on vs. the overall
population, and the supplementary business-question chapters (Chapter 6.3) each include a
segment-level breakdown of their main metric.

## Notebooks

The notebooks are the evidence trail behind the production code, not a separate exercise.
Every one of them is executed against the real pipeline data/code (via `inspect.getsource`
in `02_` and `03_`) rather than hand-copied snippets, so re-running a notebook after a code
change shows the new behavior.

- `notebooks/01_eda.ipynb` - exploratory validation of the raw dataset against the
  assumptions in `IMPLEMENTATION_PLAN (1).md`.
- `notebooks/02_blank_status_investigation.ipynb` - deep dive on the single largest finding
  (33.6% of transactions have a blank status): rules out time/payment-method/MCC/value
  artifacts, lays out why it's modeled as a third `pending` status instead of being dropped
  or reclassified, and traces that decision live into `standardize.py`, `consistency.py`,
  `transaction_level.py`, the DQ report, the PDF's limitations text, and the segmentation
  output.
- `notebooks/03_user_segmentation.ipynb` - feature selection, scaling, k selection
  (elbow + silhouette), and cluster profiling/visualization behind `src/segmentation/`.

## Testing

```bash
uv run pytest
```

Tests use small synthetic fixtures and don't need the original dataset (the end-to-end
smoke test generates its own sample CSV, but reuses the already-cached boundary file in
`data/external/` to avoid a network call).

## Known limitations

- The dataset is synthetic: 33.6% of transactions have a blank status (treated as a third
  "pending" category rather than as missing data), and most cross-variable relationships
  tested in the report are weak or not statistically significant.
- There is no promo-source column, so merchant- vs. platform-funded discounts
  (supplementary business question 2) are inferred heuristically.
- Every requested demographic attribute except Industry of Employment has a heuristic proxy
  (`src/features/demographics.py`) with its own confidence column; confidence is mostly
  "low"/"none" by design because the underlying signals are weak - see Chapter 3.10 of the
  PDF report or `data/processed/demographic_feasibility.csv` for the one attribute left
  undone and why.
- Segmentation is K-Means on a fixed feature set chosen for interpretability, not an
  exhaustive model search; silhouette scores on this dataset are modest, consistent with
  the weak correlation structure already observed in the Big Five proxy validation.
