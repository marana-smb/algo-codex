# Regime Model Pipeline Pass 1 Review

## Files created

- `models/regime_model/config.py`
- `models/regime_model/io.py`
- `models/regime_model/validate.py`
- `models/regime_model/split.py`
- `models/regime_model/export.py`
- `models/regime_model/pipeline.py`
- `notebooks/4 - regime_model pipeline orchestration v00.ipynb`

## Extracted responsibilities by module

- `config.py`
  - central run configuration for the current regime-model workflow
  - notebook-friendly toggles and naming controls
- `io.py`
  - repo-relative path resolution using `src.paths`
  - directory creation for intermediate, tables, figures, and model artifacts
- `validate.py`
  - reusable file, dataframe, column, split, and length checks
- `split.py`
  - current chronological break-date split semantics packaged into reusable helpers
- `export.py`
  - deterministic parquet, json, pickle, and artifact bundle writes
- `pipeline.py`
  - dataset-stage loading from existing reviewed intermediates
  - notebook-2-style feature list construction and training orchestration
  - saving scored partitions and model bundle outputs

## Open issues

- Raw notebook-1 feature engineering has not been extracted yet.
- `build_feature_dataset()` currently orchestrates reviewed intermediate artifacts, not raw CSV-to-feature generation.
- The training stage preserves the fast tuner path and omits dashboard/plot extraction for now.
- The notebook title says `long`, while the reviewed workflow still uses `long_short = -1` by default.

## Logic intentionally deferred

- full raw-data import and feature-engineering extraction from notebook 1
- dashboard generation
- notebook 2 visualization sections
- full-grid-search branch from notebook 2
- broader artifact/versioning policy
- generalized multi-model framework concerns

## Recommended scope for pass 2

- extract the deterministic notebook-1 feature-building path behind `build_feature_dataset()`
- move more of notebook 2 feature-selection logic into smaller internal helpers or dedicated modules
- add lightweight figure export wrappers only after training behavior is stable
- decide whether the fast tuner or full grid-search path is the canonical training path
