"""End-to-end pipeline orchestration for the regime model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.paths import DATA_INTERMEDIATE, DATA_RAW, FIGURES, MODELS, TABLES

from .features import prepare_feature_frame, select_feature_columns, summarize_data_quality
from .target import create_regime_target, drop_unusable_rows, validate_feature_columns
from .train import (
    create_chronological_splits,
    export_feature_importance_plot,
    export_model_artifacts,
    export_rfecv_plot,
    fit_with_early_stopping,
    render_roc_curve,
    run_rfecv,
    score_split,
    tune_xgb_time_series,
)


def _json_default(value):
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _build_round_paths(model_round: str) -> Dict[str, Path]:
    model = "regime_model"
    table_dir = TABLES / model / model_round
    csv_dir = table_dir / "csv"
    figure_dir = FIGURES / model / model_round
    model_dir = MODELS / model / model_round
    intermediate_dir = DATA_INTERMEDIATE / model / model_round

    paths = {
        "input_dir": DATA_RAW / "xls" / "input" / model / model_round,
        "inventory_path": DATA_RAW / "xls" / "input" / model / "master_variable_inventory 20251223.xlsx",
        "liquidity_path": DATA_RAW / "research" / "liquidity" / "xls" / "liquidity_indices_final_20260213.xlsx",
        "fear_greed_path": DATA_RAW / "research" / "fear_greed" / "csv" / "fear_and_greed_full.csv",
        "table_dir": table_dir,
        "csv_dir": csv_dir,
        "figure_dir": figure_dir,
        "model_dir": model_dir,
        "intermediate_dir": intermediate_dir,
    }
    for path in [table_dir, csv_dir, figure_dir, model_dir, intermediate_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def run_pipeline(
    model_round: str = "round_1",
    random_state: int = 42,
    tuning_iterations: int = 20,
    cv_splits: int = 3,
    rfecv_step: int = 1,
    min_features_to_select: int = 2,
    max_rows: int = None,
) -> Dict[str, object]:
    """Run the full regime model training pipeline."""
    paths = _build_round_paths(model_round)

    featured_df, master_df, optimization_df = prepare_feature_frame(
        paths["input_dir"],
        inventory_path=paths["inventory_path"],
        liquidity_path=paths["liquidity_path"],
        fear_greed_path=paths["fear_greed_path"],
    )
    targeted_df = create_regime_target(featured_df)

    feature_columns = validate_feature_columns(select_feature_columns(targeted_df, optimization_df))
    model_df = drop_unusable_rows(targeted_df, feature_columns, target_column="target_down")
    if max_rows is not None:
        model_df = model_df.tail(int(max_rows)).reset_index(drop=True)

    data_quality = summarize_data_quality(model_df[feature_columns + ["target_down"]], df_name="regime_model")
    data_quality.to_csv(str(paths["csv_dir"] / "data_quality_summary.csv"), index=True)

    splits, split_info = create_chronological_splits(model_df)
    X_train = splits["train"][feature_columns]
    y_train = splits["train"]["target_down"]
    X_test = splits["test"][feature_columns]
    y_test = splits["test"]["target_down"]
    X_oos = splits["oos"][feature_columns]
    y_oos = splits["oos"]["target_down"]

    best_params, trials_df = tune_xgb_time_series(
        X_train,
        y_train,
        n_iter=tuning_iterations,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    trials_df.to_csv(str(paths["csv_dir"] / "hyperparameter_trials.csv"), index=False)

    selected_features, rfecv = run_rfecv(
        X_train,
        y_train,
        params=best_params,
        cv_splits=cv_splits,
        step=rfecv_step,
        min_features_to_select=min_features_to_select,
        random_state=random_state,
    )

    final_model = fit_with_early_stopping(
        X_train[selected_features],
        y_train,
        params=best_params,
        random_state=random_state,
    )

    train_scores = score_split(final_model, X_train[selected_features], y_train)
    test_scores = score_split(final_model, X_test[selected_features], y_test)
    oos_scores = score_split(final_model, X_oos[selected_features], y_oos)

    scored_splits = {
        "train": splits["train"].copy(),
        "test": splits["test"].copy(),
        "oos": splits["oos"].copy(),
    }
    scored_splits["train"]["regime_probability"] = train_scores["probabilities"]
    scored_splits["test"]["regime_probability"] = test_scores["probabilities"]
    scored_splits["oos"]["regime_probability"] = oos_scores["probabilities"]

    for split_name, split_df in scored_splits.items():
        split_df.to_csv(str(paths["csv_dir"] / ("%s_scored.csv" % split_name)), index=False)
        try:
            split_df.to_parquet(str(paths["intermediate_dir"] / ("%s_scored.parquet" % split_name)), index=False)
        except Exception:
            pass

    render_roc_curve(
        y_train,
        train_scores["probabilities"],
        y_test,
        test_scores["probabilities"],
        paths["figure_dir"] / "roc_curve.png",
    )
    export_rfecv_plot(rfecv, X_train[selected_features], paths["figure_dir"] / "rfecv_curve.png")
    export_feature_importance_plot(final_model, selected_features, paths["figure_dir"] / "feature_importance.png")

    metrics = {
        "split_info": split_info,
        "selected_features": selected_features,
        "best_params": best_params,
        "train_metrics": train_scores["metrics"],
        "test_metrics": test_scores["metrics"],
        "oos_metrics": oos_scores["metrics"],
        "row_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "date_ranges": {
            name: {
                "start": frame["normed_date"].min().strftime("%Y-%m-%d"),
                "end": frame["normed_date"].max().strftime("%Y-%m-%d"),
            }
            for name, frame in splits.items()
        },
    }
    _write_json(paths["table_dir"] / "evaluation_metrics.json", metrics)

    metadata = {
        "model_name": "regime_model",
        "model_round": model_round,
        "target_column": "target_down",
        "probability_column": "regime_probability",
        "feature_columns": selected_features,
        "best_params": best_params,
        "split_info": split_info,
        "row_counts": metrics["row_counts"],
        "inventory_path": str(paths["inventory_path"]),
        "inventory_master_rows": int(len(master_df)),
        "inventory_optimization_rows": int(len(optimization_df)),
    }
    export_model_artifacts(
        model=final_model,
        feature_columns=selected_features,
        metadata=metadata,
        model_dir=paths["model_dir"],
    )

    return {
        "paths": paths,
        "metrics": metrics,
        "model": final_model,
        "feature_columns": selected_features,
    }
