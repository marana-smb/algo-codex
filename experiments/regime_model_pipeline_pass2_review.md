# Regime Model Pipeline Pass 2 Review

## Files created

- `models/regime_model/ingest.py`
- `models/regime_model/merge_external.py`
- `models/regime_model/features.py`
- `experiments/regime_model_pipeline_pass2_review.md`

## Notebook-1 logic extracted by responsibility

- `ingest.py`
  - data dictionary generation and enrichment
  - master-inventory-driven column group resolution
  - raw event-file loading through `event_import`
  - dashboard handoff into `pnl_symboldate_000`
- `features.py`
  - reviewed notebook-1 feature engineering on `static_rvw_001`
  - missing-column summary and reviewed drop-all-missing-columns behavior
- `merge_external.py`
  - liquidity merge with reviewed date normalization and merge key behavior
  - fear/greed merge with reviewed date normalization and merge key behavior
- `pipeline.py`
  - raw rebuild branch for `build_feature_dataset(config)`
  - reviewed partition creation through the split module
  - optional intermediate parquet export and dataset-stage summary writing

## Open issues

- Dictionary helper functions still write the reviewed Excel artifacts to the intermediate folder as a side effect, matching notebook behavior.
- The raw rebuild path currently emits fragmentation/performance warnings during feature-column creation because it preserves the notebook’s sequential insert style.
- The training path is intentionally left structurally unchanged in this pass.

## Behavior intentionally preserved

- raw event import and dashboard handoff semantics
- master inventory role in defining grouped import columns and distance metadata
- liquidity and fear/greed merge behavior
- notebook-1 feature definitions and sequencing
- drop-columns-with-missing-values behavior before partitioning
- reviewed chronological break-date partition logic
- existing partition names and intermediate parquet namespace

## Known differences vs notebook 1

- The pipeline writes a stable `data_review_pipeline.csv` name instead of the notebook’s date-stamped CSV when `save_tables=True`.
- Decile-analysis and exploratory summary cells were not extracted because they are review/EDA logic, not required for the dataset-stage artifact build.
- The thin orchestration notebook now explains the raw rebuild toggle explicitly.

## Recommended scope for pass 3

- decide whether to keep or reduce the reviewed dictionary-artifact side effects in the raw rebuild path
- compare raw-rebuilt outputs versus reviewed parquet artifacts for exact parity checks
- factor the notebook-2 feature-selection helpers more cleanly once dataset parity is signed off
- add targeted regression checks for row counts, date ranges, and dropped-column sets
