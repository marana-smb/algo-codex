from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .config import RunConfig
from .io import RegimeModelPaths
from .validate import require_export_ready


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _ensure_write_target(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing artifact: %s" % path)


def limit_rows(df: pd.DataFrame, debug_rows):
    if debug_rows is None:
        return df
    return df.head(int(debug_rows))


def write_parquet(df: pd.DataFrame, path: Path, *, overwrite: bool = True, debug_rows=None) -> Path:
    require_export_ready(df, df.columns.tolist()[:1], path.stem)
    _ensure_write_target(path, overwrite=overwrite)
    limit_rows(df, debug_rows).to_parquet(
        path,
        engine="pyarrow",
        index=False,
        compression="zstd",
    )
    return path


def load_partition_bundle(paths: RegimeModelPaths, partition_names: Iterable[str]) -> Dict[str, pd.DataFrame]:
    loaded = {}
    for name in partition_names:
        path = paths.intermediate_dir / ("%s.parquet" % name)
        loaded[name] = pd.read_parquet(path, engine="pyarrow")
    return loaded


def write_partition_bundle(
    partitions: Dict[str, pd.DataFrame],
    *,
    paths: RegimeModelPaths,
    config: RunConfig,
    scored: bool = False,
) -> Dict[str, Path]:
    written = {}
    target_map = paths.scored_partitions if scored else paths.intermediate_partitions
    for name, df in partitions.items():
        output_path = target_map[name]
        written[name] = write_parquet(
            df,
            output_path,
            overwrite=config.overwrite_outputs,
            debug_rows=config.debug_rows,
        )
    return written


def write_json(payload, path: Path, *, overwrite: bool = True) -> Path:
    _ensure_write_target(path, overwrite=overwrite)
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def save_model_bundle(
    *,
    model,
    feature_columns,
    metadata,
    paths: RegimeModelPaths,
    config: RunConfig,
) -> Dict[str, Path]:
    outputs = {}

    _ensure_write_target(paths.model_artifact_path, overwrite=config.overwrite_outputs)
    with open(str(paths.model_artifact_path), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outputs["model"] = paths.model_artifact_path

    if config.save_feature_columns:
        _ensure_write_target(paths.feature_columns_path, overwrite=config.overwrite_outputs)
        with open(str(paths.feature_columns_path), "wb") as handle:
            pickle.dump(list(feature_columns), handle, protocol=pickle.HIGHEST_PROTOCOL)
        outputs["feature_columns"] = paths.feature_columns_path

    outputs["metadata"] = write_json(
        metadata,
        paths.metadata_path,
        overwrite=config.overwrite_outputs,
    )
    return outputs
