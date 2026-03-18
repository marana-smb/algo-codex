from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from src.paths import DATA_INTERMEDIATE, DATA_RAW, FIGURES, MODELS, TABLES

if TYPE_CHECKING:
    from .config import RunConfig


@dataclass
class RegimeModelPaths:
    raw_input_dir: Path
    raw_input_files: Dict[str, Path]
    master_inventory_path: Path
    liquidity_path: Path
    fear_greed_path: Path
    intermediate_dir: Path
    intermediate_partitions: Dict[str, Path]
    scored_partitions: Dict[str, Path]
    table_dir: Path
    table_csv_dir: Path
    table_xls_dir: Path
    figure_dir: Path
    model_dir: Path
    model_artifact_path: Path
    feature_columns_path: Path
    metadata_path: Path
    dataset_summary_path: Path
    training_summary_path: Path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_input_paths(config: "RunConfig") -> Dict[str, Path]:
    raw_input_dir = DATA_RAW / "xls" / "input" / config.model_name / config.round_name
    return {filename: raw_input_dir / filename for filename in config.raw_event_filenames}


def get_external_input_paths(config: "RunConfig") -> Dict[str, Path]:
    return {
        "master_inventory": DATA_RAW / "xls" / "input" / config.model_name / config.round_name / config.master_data_filename,
        "liquidity": DATA_RAW / "research" / "liquidity" / "xls" / config.liquidity_filename,
        "fear_greed": DATA_RAW / "research" / "fear_greed" / "csv" / config.fear_greed_filename,
    }


def get_intermediate_paths(config: "RunConfig") -> Dict[str, Path]:
    intermediate_dir = DATA_INTERMEDIATE / config.model_name / config.round_name
    _ensure_dir(intermediate_dir)
    return {
        name: intermediate_dir / ("%s.parquet" % name)
        for name in config.intermediate_partition_names
    }


def get_output_paths(config: "RunConfig") -> Dict[str, Path]:
    table_dir = _ensure_dir(TABLES / config.model_name / config.round_name)
    table_csv_dir = _ensure_dir(table_dir / "csv")
    table_xls_dir = _ensure_dir(table_dir / "xls")
    figure_dir = _ensure_dir(FIGURES / config.model_name / config.round_name)
    model_dir = _ensure_dir(MODELS / config.model_name / config.round_name)
    stem = config.model_artifact_stem
    return {
        "table_dir": table_dir,
        "table_csv_dir": table_csv_dir,
        "table_xls_dir": table_xls_dir,
        "figure_dir": figure_dir,
        "model_dir": model_dir,
        "model_artifact_path": model_dir / ("%s.pkl" % stem),
        "feature_columns_path": model_dir / ("%s_feature_columns.pkl" % stem),
        "metadata_path": model_dir / ("%s_metadata.json" % stem),
        "dataset_summary_path": table_dir / "dataset_stage_summary.json",
        "training_summary_path": table_dir / "training_stage_summary.json",
    }


def get_model_paths(config: "RunConfig") -> RegimeModelPaths:
    raw_input_dir = DATA_RAW / "xls" / "input" / config.model_name / config.round_name
    raw_inputs = get_raw_input_paths(config)
    external_inputs = get_external_input_paths(config)
    intermediate_dir = _ensure_dir(DATA_INTERMEDIATE / config.model_name / config.round_name)
    output_paths = get_output_paths(config)
    intermediate_partitions = get_intermediate_paths(config)
    scored_partitions = {
        name: intermediate_dir / ("%s.parquet" % name)
        for name in config.scored_partition_names
    }

    return RegimeModelPaths(
        raw_input_dir=raw_input_dir,
        raw_input_files=raw_inputs,
        master_inventory_path=external_inputs["master_inventory"],
        liquidity_path=external_inputs["liquidity"],
        fear_greed_path=external_inputs["fear_greed"],
        intermediate_dir=intermediate_dir,
        intermediate_partitions=intermediate_partitions,
        scored_partitions=scored_partitions,
        table_dir=output_paths["table_dir"],
        table_csv_dir=output_paths["table_csv_dir"],
        table_xls_dir=output_paths["table_xls_dir"],
        figure_dir=output_paths["figure_dir"],
        model_dir=output_paths["model_dir"],
        model_artifact_path=output_paths["model_artifact_path"],
        feature_columns_path=output_paths["feature_columns_path"],
        metadata_path=output_paths["metadata_path"],
        dataset_summary_path=output_paths["dataset_summary_path"],
        training_summary_path=output_paths["training_summary_path"],
    )
