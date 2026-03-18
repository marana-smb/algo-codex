from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from notebook_utils._grid_search_37_v2 import tune_xgb_fast_compatible

from .config import RunConfig
from .export import load_partition_bundle, save_model_bundle, write_json, write_partition_bundle, write_parquet
from .features import drop_missing_feature_columns, engineer_reviewed_features, summarize_missing_columns
from .ingest import build_dashboard_dataset, build_inventory_dataframes, load_raw_event_data, resolve_column_groups
from .io import RegimeModelPaths, get_model_paths
from .merge_external import merge_fear_greed, merge_liquidity_features
from .split import create_split_bundle
from .validate import (
    require_columns,
    require_file,
    require_files,
    require_matching_lengths,
    require_non_empty_df,
    require_non_empty_list,
)


GROUP_ALWAYS = [
    "atr_250",
    "atr_ema_a",
    "atr_ema_b",
    "atr_ktg",
    "atr_sma_a",
    "atr_sma_b",
    "beta",
    "cmf",
    "corr_1y",
    "corr_20d",
    "d_atr",
    "d_avol5",
    "d_avol50",
    "d_natr",
    "d_natr_ktg",
    "d_rsi",
    "kalmar_q",
    "pct_chg_open",
    "rsi",
    "rvol",
    "vol_20d",
    "vol_5d",
    "vol_60d",
    "spy_atr",
    "spy_rvol",
    "PCA_Index_ma5",
    "PCA_ScaledIndex_ma5",
    "PCA_Index_ma20",
    "PCA_ScaledIndex_ma20",
    "PCA_Index_ma50",
    "PCA_ScaledIndex_ma50",
    "PCA_Raw_full",
    "PCA_Index_full",
    "fear_greed",
]

GROUP_CALENDAR = [
    "entry_hr_dec",
    "entry_hr_dec_to_close",
    "week_day_sin",
    "week_day_cos",
    "month_sin",
    "month_cos",
    "year_day_sin",
    "year_day_cos",
]

EXCLUDE_COLUMNS = {
    "mtm_pl",
    "pl_g",
    "pl_n",
    "fees",
    "Capital",
    "wins",
    "target_return",
    "target_down",
    "source_file",
    "entry_time",
    "exit_time",
    "normed_date",
    "symbol",
}

FEATURE_RULES = {
    "distance": lambda c: c.startswith("dist_"),
    "percent": lambda c: c.startswith("pct_"),
    "return": lambda c: c.startswith("ret_"),
    "dummy": lambda c: c.startswith("dumm_"),
    "volume_ratio": lambda c: ("vol" in c and "_rat" in c),
}


@dataclass
class DatasetStageResult:
    config: RunConfig
    paths: RegimeModelPaths
    partitions: Dict[str, pd.DataFrame]
    source: str
    row_counts: Dict[str, int]
    column_counts: Dict[str, int]
    date_ranges: Dict[str, Dict[str, Optional[str]]]


@dataclass
class TrainStageResult:
    config: RunConfig
    paths: RegimeModelPaths
    model: object
    all_feature_columns: List[str]
    selected_feature_columns: List[str]
    rfecv_feature_columns: List[str]
    best_params: Dict[str, object]
    metrics: Dict[str, object]
    scored_partitions: Dict[str, pd.DataFrame]
    artifact_paths: Dict[str, Path]


def _date_range(df: pd.DataFrame, date_col: str = "normed_date") -> Dict[str, Optional[str]]:
    if df.empty:
        return {"start": None, "end": None}
    dates = pd.to_datetime(df[date_col])
    return {
        "start": dates.min().strftime("%Y-%m-%d"),
        "end": dates.max().strftime("%Y-%m-%d"),
    }


def _summarize_partitions(partitions: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    return {
        "row_counts": {name: int(len(df)) for name, df in partitions.items()},
        "column_counts": {name: int(df.shape[1]) for name, df in partitions.items()},
        "date_ranges": {name: _date_range(df) for name, df in partitions.items()},
    }


def build_feature_list_from_columns(df: pd.DataFrame) -> List[str]:
    cols = df.columns.tolist()
    selected = []

    for col in GROUP_ALWAYS:
        if col in cols:
            selected.append(col)

    for col in GROUP_CALENDAR:
        if col in cols:
            selected.append(col)

    for _, rule in FEATURE_RULES.items():
        selected.extend([col for col in cols if rule(col)])

    selected = list(dict.fromkeys(selected))
    selected = [col for col in selected if col not in EXCLUDE_COLUMNS]
    require_non_empty_list(selected, "feature_columns")
    return selected


def compute_rank_ic_stats(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "wins",
    date_col: str = "normed_date",
) -> pd.DataFrame:
    grouped = df.groupby(date_col)
    rows = []

    for feature in features:
        daily_ic = grouped.apply(lambda group: group[feature].corr(group[target_col], method="spearman"))
        ic_std = daily_ic.std()
        rows.append(
            {
                "feature": feature,
                "IC_mean": daily_ic.mean(),
                "IC_std": ic_std,
                "IC_IR": daily_ic.mean() / ic_std if ic_std and not np.isnan(ic_std) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("IC_mean", ascending=False)


def prune_correlated_features_with_summary(
    df: pd.DataFrame,
    ranked_features: List[str],
    corr_threshold: float = 0.85,
    method: str = "pearson",
):
    ranked_features = [feature for feature in ranked_features if feature in df.columns]
    require_non_empty_list(ranked_features, "ranked_features")

    corr_matrix = df[ranked_features].corr(method=method).abs()
    selected_features = []
    dropped_features = set()
    dropped_map = {}

    for feature in ranked_features:
        if feature in dropped_features:
            continue

        selected_features.append(feature)
        correlated_cluster = corr_matrix.index[corr_matrix.loc[feature] > corr_threshold].tolist()
        to_drop = [item for item in correlated_cluster if item != feature]
        dropped_map[feature] = to_drop
        for item in to_drop:
            dropped_features.add(item)

    summary_rows = [
        {
            "keeper": keeper,
            "n_dropped": len(dropped),
            "dropped_features": ", ".join(dropped),
        }
        for keeper, dropped in dropped_map.items()
    ]
    cluster_summary = pd.DataFrame(summary_rows).sort_values(["n_dropped", "keeper"], ascending=[False, True])
    return selected_features, dropped_map, corr_matrix, cluster_summary


def _select_training_features(partition_ins_80: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
    variance_filter = partition_ins_80[feature_columns].var()
    optmz_list_all = variance_filter[variance_filter > 1e-6].index.tolist()
    require_non_empty_list(optmz_list_all, "optmz_list_all")

    corr_with_pl = (
        partition_ins_80[optmz_list_all]
        .corrwith(partition_ins_80["mtm_pl"])
        .abs()
        .sort_values(ascending=False)
    )
    top_features_pl = corr_with_pl.head(50).index.tolist()
    corr_matrix_pl = partition_ins_80[top_features_pl].corr().abs()

    corr_mtm_list = []
    dropped_features = set()
    for col in top_features_pl:
        if col in dropped_features:
            continue
        corr_mtm_list.append(col)
        correlated = corr_matrix_pl.index[corr_matrix_pl[col] > 0.85].tolist()
        for feat in correlated:
            if feat != col:
                dropped_features.add(feat)

    corr_with_wins = (
        partition_ins_80[optmz_list_all]
        .corrwith(partition_ins_80["wins"])
        .abs()
        .sort_values(ascending=False)
    )
    top_features_wins = corr_with_wins.head(50).index.tolist()
    corr_matrix_wins = partition_ins_80[top_features_wins].corr().abs()

    corr_wins_list = []
    dropped_features = set()
    for col in top_features_wins:
        if col in dropped_features:
            continue
        corr_wins_list.append(col)
        correlated = corr_matrix_wins.index[corr_matrix_wins[col] > 0.85].tolist()
        for feat in correlated:
            if feat != col:
                dropped_features.add(feat)

    ic_table = compute_rank_ic_stats(partition_ins_80, optmz_list_all, target_col="wins", date_col="normed_date")
    ic_top_list = ic_table.sort_values("IC_IR", ascending=False).head(50)["feature"].tolist()
    ic_pruned_list, dropped_map, _, cluster_summary = prune_correlated_features_with_summary(
        df=partition_ins_80,
        ranked_features=ic_top_list,
        corr_threshold=0.85,
        method="pearson",
    )

    all_features = corr_mtm_list + corr_wins_list + ic_pruned_list
    feature_votes = Counter(all_features)
    selected_features = [feature for feature, votes in feature_votes.items() if votes >= 2]
    require_non_empty_list(selected_features, "selected_features")

    return {
        "feature_columns": feature_columns,
        "variance_filtered_columns": optmz_list_all,
        "corr_mtm_list": corr_mtm_list,
        "corr_wins_list": corr_wins_list,
        "ic_pruned_list": ic_pruned_list,
        "selected_features": selected_features,
        "cluster_summary": cluster_summary,
        "dropped_map": dropped_map,
    }


def _calculate_max_train_size(df: pd.DataFrame, target_lookback_months: int) -> int:
    indexed = df.copy()
    indexed["normed_date"] = pd.to_datetime(indexed["normed_date"])
    indexed = indexed.set_index("normed_date")
    daily_counts = indexed.resample("1D").size()
    active_days = daily_counts[daily_counts > 0]
    require_non_empty_df(active_days.to_frame("rows"), "active_days")
    avg_rows_per_day = active_days.mean()
    target_lookback_days = target_lookback_months * 21
    calculated_size = int(avg_rows_per_day * target_lookback_days)
    if calculated_size <= 0:
        raise ValueError("Calculated max_train_size must be positive, got %s." % calculated_size)
    return calculated_size


def _classification_summary(y_true, pred) -> Dict[str, object]:
    return {
        "classification_report": classification_report(y_true, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def build_feature_dataset(config: RunConfig) -> DatasetStageResult:
    paths = get_model_paths(config)

    if config.use_existing_intermediate:
        require_files(paths.intermediate_partitions.values(), "intermediate partition")
        partitions = load_partition_bundle(paths, config.intermediate_partition_names)
        for name, df in partitions.items():
            require_non_empty_df(df, name)
            require_columns(df, ["normed_date", config.target_column, "mtm_pl"], name)

        summary = _summarize_partitions(partitions)
        result = DatasetStageResult(
            config=config,
            paths=paths,
            partitions=partitions,
            source="existing_intermediate",
            row_counts=summary["row_counts"],
            column_counts=summary["column_counts"],
            date_ranges=summary["date_ranges"],
        )

        if config.save_tables:
            write_json(
                {
                    "source": result.source,
                    "row_counts": result.row_counts,
                    "column_counts": result.column_counts,
                    "date_ranges": result.date_ranges,
                },
                paths.dataset_summary_path,
                overwrite=config.overwrite_outputs,
            )
        return result

    for raw_path in paths.raw_input_files.values():
        require_file(raw_path, "raw event input")
    require_file(paths.master_inventory_path, "master inventory")
    require_file(paths.liquidity_path, "liquidity input")
    require_file(paths.fear_greed_path, "fear and greed input")

    _, final_df = build_inventory_dataframes(config, paths)
    grouped_columns = resolve_column_groups(final_df)
    event_data = load_raw_event_data(config, paths, grouped_columns)
    pnl_symboldate_000, _ = build_dashboard_dataset(config, event_data)

    static_rvw_001 = engineer_reviewed_features(pnl_symboldate_000, final_df)
    static_rvw_002 = merge_liquidity_features(static_rvw_001, paths)
    static_rvw_003 = merge_fear_greed(static_rvw_002, paths)

    summary, missing_list = summarize_missing_columns(static_rvw_003)
    partition_a = drop_missing_feature_columns(static_rvw_003, missing_list)


    # ADDED:--- SAVE FULL PRE-SPLIT DATASET (CRITICAL ARTIFACT) ---
    full_dataset_path = paths.intermediate_dir / "feature_dataset_full.parquet"

    write_parquet(
        partition_a,
        full_dataset_path,
        overwrite=config.overwrite_outputs,
        debug_rows=config.debug_rows,
    )

    print(f"[INFO] Saved feature_dataset_full to {full_dataset_path} with shape {partition_a.shape}")
    #

    require_columns(partition_a, ["normed_date", config.target_column, "mtm_pl"], "partition_a")

    split_bundle = create_split_bundle(
        partition_a,
        insample_fraction=config.insample_fraction,
        train_fraction=config.train_fraction,
        target_column=config.target_column,
        ins_start_date=config.ins_start_date,
    )
    partitions = split_bundle.as_dict()
    for name, df in partitions.items():
        require_non_empty_df(df, name)
        require_columns(df, ["normed_date", config.target_column, "mtm_pl"], name)

    if config.save_intermediate:
        write_partition_bundle(
            partitions,
            paths=paths,
            config=config,
            scored=False,
        )

    summary_payload = _summarize_partitions(partitions)
    # ADDED: -- metasummary of full data
    summary_payload["source"] = "raw_rebuilt"
    summary_payload["feature_dataset_full"] = {
        "path": str(full_dataset_path),
        "rows_in_memory": int(partition_a.shape[0]),
        "cols": int(partition_a.shape[1]),
        "debug_rows_applied": config.debug_rows,
    }
    #

    summary_payload["save_debug_rows"] = config.debug_rows
    summary_payload["missing_columns_dropped"] = missing_list
    summary_payload["review_shapes"] = {
        "event_data": [int(event_data.shape[0]), int(event_data.shape[1])],
        "pnl_symboldate_000": [int(pnl_symboldate_000.shape[0]), int(pnl_symboldate_000.shape[1])],
        "static_rvw_001": [int(static_rvw_001.shape[0]), int(static_rvw_001.shape[1])],
        "static_rvw_002": [int(static_rvw_002.shape[0]), int(static_rvw_002.shape[1])],
        "static_rvw_003": [int(static_rvw_003.shape[0]), int(static_rvw_003.shape[1])],
        "partition_a": [int(partition_a.shape[0]), int(partition_a.shape[1])],
    }

    if config.save_tables:
        write_json(
            summary_payload,
            paths.dataset_summary_path,
            overwrite=config.overwrite_outputs,
        )
        summary[summary["missing"] == 0].to_csv(paths.intermediate_dir / "data_review_pipeline.csv", index=True)

    return DatasetStageResult(
        config=config,
        paths=paths,
        partitions=partitions,
        source="raw_rebuilt",
        row_counts=summary_payload["row_counts"],
        column_counts=summary_payload["column_counts"],
        date_ranges=summary_payload["date_ranges"],
    )


def train_regime_xgb(config: RunConfig, dataset: Optional[DatasetStageResult] = None) -> TrainStageResult:
    dataset_result = dataset if dataset is not None else build_feature_dataset(config)
    partitions = dataset_result.partitions
    paths = dataset_result.paths

    partition_ins_80 = partitions["partition_ins_80"]
    partition_ins_20 = partitions["partition_ins_20"]
    partition_oos = partitions["partition_oos"]

    # ADDED - uses debug_rows to create a subset of the full data to run test
    if config.debug_rows is not None:
        debug_n = config.debug_rows
        partition_ins_80 = partition_ins_80.head(min(config.debug_rows, len(partition_ins_80))).copy()
        partition_ins_20 = partition_ins_20.head(min(config.debug_rows, len(partition_ins_20))).copy()
        partition_oos = partition_oos.head(min(config.debug_rows, len(partition_oos))).copy()

        partitions = {
            **partitions,
            "partition_ins_80": partition_ins_80,
            "partition_ins_20": partition_ins_20,
            "partition_oos": partition_oos,
            "partition_ins": pd.concat([partition_ins_80, partition_ins_20], axis=0).reset_index(drop=True),
        }
    #

    for name, df in partitions.items():
        require_non_empty_df(df, name)
        require_columns(df, ["normed_date", config.target_column, "mtm_pl"], name)

    all_feature_columns = build_feature_list_from_columns(partition_ins_80)
    selection = _select_training_features(partition_ins_80, all_feature_columns)
    selected_feature_columns = selection["selected_features"]
    ml_list = selected_feature_columns if config.use_selected_features else all_feature_columns
    require_non_empty_list(ml_list, "ml_list")

    for df_name, df in (
        ("partition_ins_80", partition_ins_80),
        ("partition_ins_20", partition_ins_20),
        ("partition_oos", partition_oos),
    ):
        require_columns(df, ml_list + [config.target_column], df_name)

    X_train = partition_ins_80[ml_list]
    y_train = partition_ins_80[config.target_column]
    X_test = partition_ins_20[ml_list]
    y_test = partition_ins_20[config.target_column]
    X_oos = partition_oos[ml_list]
    y_oos = partition_oos[config.target_column]

    model, best_params = tune_xgb_fast_compatible(
        X_train,
        y_train,
        X_test,
        y_test,
        search_rows=config.search_rows,
        n_iter=config.n_iter,
        cv_splits=config.cv_splits,
        n_jobs=config.search_n_jobs,
        search_n_estimators=config.search_n_estimators,
        final_n_estimators_cap=config.final_n_estimators_cap,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.random_state,
    )

    max_train_size = _calculate_max_train_size(partition_ins_80, config.target_lookback_months)
    rfecv = RFECV(
        estimator=model,
        step=config.rfecv_step,
        min_features_to_select=config.rfecv_min_features_to_select,
        cv=TimeSeriesSplit(max_train_size=max_train_size, n_splits=config.cv_splits),
        scoring=config.rfecv_scoring,
        n_jobs=config.rfecv_n_jobs,
    )
    rfecv.fit(X_train, y_train)

    rfecv_feature_columns = X_train.columns[rfecv.support_].tolist()
    require_non_empty_list(rfecv_feature_columns, "rfecv_feature_columns")

    X_train_rfecv = partition_ins_80[rfecv_feature_columns]
    X_test_rfecv = partition_ins_20[rfecv_feature_columns]
    X_oos_rfecv = partition_oos[rfecv_feature_columns]

    model.fit(X_train_rfecv, y_train)

    yhat_train_1 = model.predict_proba(X_train_rfecv)[:, 1]
    yhat_test_1 = model.predict_proba(X_test_rfecv)[:, 1]
    yhat_oos_1 = model.predict_proba(X_oos_rfecv)[:, 1]

    # ADDED: including prediction threshold
    threshold = config.classification_threshold

    pred_train = (yhat_train_1 >= threshold).astype(int)
    pred_test = (yhat_test_1 >= threshold).astype(int)
    pred_oos = (yhat_oos_1 >= threshold).astype(int)
    #

    require_matching_lengths(len(partition_ins_80), len(yhat_train_1), "train predictions")
    require_matching_lengths(len(partition_ins_20), len(yhat_test_1), "test predictions")
    require_matching_lengths(len(partition_oos), len(yhat_oos_1), "oos predictions")

    scored_partitions = {
        "partition_ins_80_001": partition_ins_80.copy(),
        "partition_ins_20_001": partition_ins_20.copy(),
        "partition_oos_001": partition_oos.copy(),
    }
    scored_partitions["partition_ins_80_001"][config.probability_column] = yhat_train_1
    scored_partitions["partition_ins_20_001"][config.probability_column] = yhat_test_1
    scored_partitions["partition_oos_001"][config.probability_column] = yhat_oos_1

    artifact_paths = {}
    if config.save_intermediate:
        artifact_paths.update(
            write_partition_bundle(
                scored_partitions,
                paths=paths,
                config=config,
                scored=True,
            )
        )

    metrics = {
        "best_params": best_params,
        "max_train_size": max_train_size,
        "all_feature_count": len(all_feature_columns),
        "selected_feature_count": len(selected_feature_columns),
        "rfecv_feature_count": len(rfecv_feature_columns),
        "train": _classification_summary(y_train, pred_train),
        "test": _classification_summary(y_test, pred_test),
        "oos": _classification_summary(y_oos, pred_oos),
        #ADDED
        "threshold": config.classification_threshold,
        "probability_means": {
            "train": float(np.mean(yhat_train_1)),
            "test": float(np.mean(yhat_test_1)),
            "oos": float(np.mean(yhat_oos_1)),
        },
        "feature_selection": {
            "corr_mtm_list": selection["corr_mtm_list"],
            "corr_wins_list": selection["corr_wins_list"],
            "ic_pruned_list": selection["ic_pruned_list"],
            "selected_features": selected_feature_columns,
            "rfecv_features": rfecv_feature_columns,
        },
    }

    if config.save_model:
        artifact_paths.update(
            save_model_bundle(
                model=model,
                feature_columns=rfecv_feature_columns,
                metadata={
                    "model_name": config.model_name,
                    "round_name": config.round_name,
                    "side_label": config.side_label,
                    "target_column": config.target_column,
                    "probability_column": config.probability_column,
                    "best_params": best_params,
                    "selected_feature_columns": selected_feature_columns,
                    "rfecv_feature_columns": rfecv_feature_columns,
                    "dataset_source": dataset_result.source,
                    #ADDED
                    "classification_threshold": config.classification_threshold,
                },
                paths=paths,
                config=config,
            )
        )

    if config.save_tables:
        artifact_paths["training_summary"] = write_json(
            metrics,
            paths.training_summary_path,
            overwrite=config.overwrite_outputs,
        )

    return TrainStageResult(
        config=config,
        paths=paths,
        model=model,
        all_feature_columns=all_feature_columns,
        selected_feature_columns=selected_feature_columns,
        rfecv_feature_columns=rfecv_feature_columns,
        best_params=best_params,
        metrics=metrics,
        scored_partitions=scored_partitions,
        artifact_paths=artifact_paths,
    )


def run_full_pipeline(config: RunConfig) -> Dict[str, object]:
    dataset = build_feature_dataset(config)
    training = train_regime_xgb(config, dataset=dataset)
    return {
        "dataset": dataset,
        "training": training,
    }


__all__ = [
    "DatasetStageResult",
    "TrainStageResult",
    "build_feature_dataset",
    "train_regime_xgb",
    "run_full_pipeline",
]
