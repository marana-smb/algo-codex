from __future__ import annotations

import ast
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from notebook_utils import _fast_optimization_v1
from notebook_utils._bayesian_optimization_v1 import bayesian_greedy_threshold_search
from notebook_utils._dashboard_functions_one_symbol_v2 import dashboard
from notebook_utils._fast_optimization_v1 import greedy_threshold_search
from notebook_utils._optimization_functions_37_v2 import optmz_loop_wrap

from .config import RunConfig
from .io import get_model_paths
from .pipeline import TrainStageResult, build_feature_list_from_columns
from .validate import require_columns, require_file, require_non_empty_df


PARTITION_NAMES = (
    "partition_ins_80_001",
    "partition_ins_20_001",
    "partition_oos_001",
)

PARTITION_OUTPUT_LABELS = {
    "partition_ins_80_001": "ins_80_1",
    "partition_ins_20_001": "ins_20_1",
    "partition_oos_001": "oos_1",
}

REQUIRED_BASE_COLUMNS = [
    "normed_date",
    "symbol",
    "entry_time",
    "entry_price",
    "entry_shares",
    "exit_price",
    "exit_shares",
    "matched_shares",
    "mtm_pl",
    "entry_pl",
    "entry_side",
    "entry_fees",
    "exit_fees",
    "pl_g",
    "pl_n",
    "fees",
]

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
    "fear_greed",
]


@dataclass
class StaticOptimizationConfig:
    run_id: str = "static_opt"
    max_capital: float = 68500.0
    entry_fee: float = 3.5
    long_short: int = -1
    min_trades: int = 25
    greedy_n_quantiles: int = 24
    greedy_max_features: int = 6
    greedy_improvement_eps: float = 0.02
    search_min_count_per_side: int = 200
    round_crit_digits: int = 4
    bayesian_n_calls_per_step: int = 35
    bayesian_n_random_starts: int = 10
    bayesian_tail_penalty: float = 0.4
    bayesian_complexity_penalty: float = 0.03
    bayesian_lock_to_grid: bool = True
    bayesian_n_quantiles_lock: int = 32
    save_best_json: bool = False
    save_equity_curve: bool = False
    save_partition_dashboards_debug: bool = False
    feature_threshold_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchMethodArtifacts:
    method_name: str
    ranked_candidates: pd.DataFrame
    candidate_csv_path: Path
    best_json_path: Optional[Path]
    selected_row: Dict[str, Any]


@dataclass
class SelectedRuleReview:
    rule_name: str
    method_name: str
    candidate_name: str
    crit_list_json: str
    combined_dashboard_df: pd.DataFrame
    combined_dashboard_path: Optional[Path]
    combined_curve_df: pd.DataFrame
    combined_curve_path: Optional[Path]
    bydate_map: Dict[str, pd.DataFrame]


@dataclass
class StaticOptimizationStageResult:
    run_config: RunConfig
    static_config: StaticOptimizationConfig
    partitions: Dict[str, pd.DataFrame]
    method_results: Dict[str, SearchMethodArtifacts]
    selected_rule_reviews: Dict[str, SelectedRuleReview]
    validation_summary: pd.DataFrame
    validation_summary_path: Path

    @property
    def selected_candidate_summaries(self) -> pd.DataFrame:
        rows = []
        for review in self.selected_rule_reviews.values():
            rows.append(
                {
                    "rule_name": review.rule_name,
                    "method_name": review.method_name,
                    "candidate_name": review.candidate_name,
                    "crit_list_json": review.crit_list_json,
                }
            )
        return pd.DataFrame(rows)


def _safe_feature_subset(available_columns: List[str], candidate_columns: List[str]) -> List[str]:
    available = set(available_columns)
    return [col for col in candidate_columns if col in available]


def _ensure_partition_ready(df: pd.DataFrame, df_name: str) -> None:
    require_non_empty_df(df, df_name)
    require_columns(df, REQUIRED_BASE_COLUMNS, df_name)


def _static_output_dirs(run_config: RunConfig, static_config: StaticOptimizationConfig) -> Dict[str, Path]:
    paths = get_model_paths(run_config)
    output_dir = paths.table_dir / static_config.run_id
    csv_dir = output_dir / "csv"
    json_dir = output_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "csv_dir": csv_dir,
        "json_dir": json_dir,
    }


def _load_scored_partitions(run_config: RunConfig) -> Dict[str, pd.DataFrame]:
    paths = get_model_paths(run_config)
    loaded: Dict[str, pd.DataFrame] = {}
    for name in PARTITION_NAMES:
        path = paths.scored_partitions[name]
        require_file(path, "scored partition")
        loaded[name] = pd.read_parquet(path, engine="pyarrow")
        _ensure_partition_ready(loaded[name], name)
    return loaded


def _resolve_scored_partitions(
    run_config: RunConfig,
    training: Optional[TrainStageResult] = None,
) -> Dict[str, pd.DataFrame]:
    if training is not None and training.scored_partitions:
        partitions = {name: training.scored_partitions[name].copy() for name in PARTITION_NAMES}
        for name, df in partitions.items():
            _ensure_partition_ready(df, name)
        return partitions
    return _load_scored_partitions(run_config)


def normalize_crit_list(crit_list: Any) -> List[Dict[str, Tuple[float, str]]]:
    if crit_list is None:
        return []
    if isinstance(crit_list, str):
        crit_list = ast.literal_eval(crit_list)
    if isinstance(crit_list, dict):
        crit_list = [crit_list]

    normalized: List[Dict[str, Tuple[float, str]]] = []
    for item in crit_list:
        if not isinstance(item, dict):
            continue
        normalized_item: Dict[str, Tuple[float, str]] = {}
        for key, value in item.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                threshold, operator_name = value
                normalized_item[str(key)] = (float(threshold), str(operator_name))
        if normalized_item:
            normalized.append(normalized_item)
    return normalized


def crit_list_to_rule_dict(crit_list: Any) -> Dict[str, Tuple[float, str]]:
    crit_list = normalize_crit_list(crit_list)
    combined: Dict[str, Tuple[float, str]] = {}
    for item in crit_list:
        combined.update(item)
    return combined


def rule_dict_to_crit_list(rule_dict: Dict[str, Tuple[float, str]]) -> List[Dict[str, Tuple[float, str]]]:
    return [dict(rule_dict)] if rule_dict else []


def crit_list_to_literal(crit_list: Any) -> str:
    return repr(crit_list_to_rule_dict(crit_list))


def chosen_to_crit_list(chosen: List[Any]) -> List[Dict[str, Tuple[float, str]]]:
    rule_dict = {
        candidate.feat: (float(candidate.threshold), str(candidate.op))
        for candidate in chosen
    }
    return rule_dict_to_crit_list(rule_dict)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _save_dataframe_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _save_json(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
    return path


def _extract_dashboard_metrics(strat_df: pd.DataFrame) -> Dict[str, Any]:
    if strat_df is None or strat_df.empty:
        return {}
    metrics = strat_df.copy()
    metrics["metric"] = metrics["metric"].astype(str)
    return dict(zip(metrics["metric"], metrics["value"]))


def _is_monotonic_equity_curve(bydate_df: pd.DataFrame, curve_col: str = "cum_pl_n") -> bool:
    if bydate_df is None or bydate_df.empty or curve_col not in bydate_df.columns:
        return False
    curve = bydate_df[curve_col].dropna()
    if curve.empty:
        return False
    increments = curve.diff().fillna(curve.iloc[0])
    return bool((increments >= 0).all())


def round_threshold_for_feature(
    feature_name: str,
    value: float,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    overrides = overrides or {}
    if feature_name in overrides:
        override = overrides[feature_name]
        return override(value) if callable(override) else round(float(value), int(override))

    feature_lower = feature_name.lower()
    value = float(value)
    abs_value = abs(value)

    if "proba" in feature_lower or "probability" in feature_lower:
        return round(value, 2)
    if feature_lower in {"fear_greed", "rsi"} or feature_lower.endswith("_rsi"):
        return int(round(value))
    if any(token in feature_lower for token in ["vol", "rvol", "atr", "natr", "beta", "cmf", "kalmar", "scaled"]):
        if abs_value < 10:
            return round(value, 2)
        if abs_value < 100:
            return round(value, 1)
        return int(round(value))
    if feature_lower.startswith("ret_") or feature_lower.startswith("pct_") or feature_lower.startswith("dist_"):
        if abs_value < 1:
            return round(value, 3)
        if abs_value < 10:
            return round(value, 2)
        return round(value, 1)
    if abs_value < 1:
        return round(value, 3)
    if abs_value < 10:
        return round(value, 2)
    if abs_value < 100:
        return round(value, 1)
    return int(round(value))


def round_crit_list(
    crit_list: Any,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Tuple[float, str]]]:
    rule_dict = crit_list_to_rule_dict(crit_list)
    rounded_rule_dict = {
        feature_name: (
            round_threshold_for_feature(feature_name, threshold, overrides=overrides),
            operator_name,
        )
        for feature_name, (threshold, operator_name) in rule_dict.items()
    }
    return rule_dict_to_crit_list(rounded_rule_dict)


def evaluate_rule_metrics(
    df: pd.DataFrame,
    crit_list: Any,
    static_config: StaticOptimizationConfig,
) -> Dict[str, Any]:
    metrics = _fast_optimization_v1.evaluate_threshold_rule(
        df,
        normalize_crit_list(crit_list),
        mtm_col="mtm_pl",
        date_col="normed_date",
        use_numexpr=True,
        crit_round_digits=static_config.round_crit_digits,
    )
    return dict(metrics)


def summarize_partition_rule(
    df: pd.DataFrame,
    crit_list: Any,
    partition_name: str,
    method_name: str,
    candidate_name: str,
    static_config: StaticOptimizationConfig,
) -> Dict[str, Any]:
    crit_list = normalize_crit_list(crit_list)
    rule_dict = crit_list_to_rule_dict(crit_list)
    missing_rule_columns = [col for col in rule_dict if col not in df.columns]
    if missing_rule_columns:
        raise KeyError(f"{partition_name} is missing rule columns: {missing_rule_columns}")

    shared_metrics = evaluate_rule_metrics(df, crit_list, static_config=static_config)
    trade_count = int(shared_metrics["trade_count"])
    sharpe = float(shared_metrics["sharpe"]) if np.isfinite(shared_metrics["sharpe"]) else np.nan
    ann_return = float(shared_metrics["ann_return"]) if np.isfinite(shared_metrics["ann_return"]) else np.nan
    ann_vol = float(shared_metrics["ann_vol"]) if np.isfinite(shared_metrics["ann_vol"]) else np.nan
    max_dd = float(shared_metrics["max_dd"]) if np.isfinite(shared_metrics["max_dd"]) else np.nan
    capital_used = float(shared_metrics["capital_used"])

    optmz_summary, optmz_events, optmz_bydate = optmz_loop_wrap(df, crit_list, static_config.max_capital)
    if isinstance(optmz_bydate, tuple):
        optmz_bydate = None

    if optmz_events is None or isinstance(optmz_events, tuple) or len(optmz_events) == 0:
        return {
            "summary_row": {
                "method_name": method_name,
                "candidate_name": candidate_name,
                "partition_name": partition_name,
                "partition_label": PARTITION_OUTPUT_LABELS[partition_name],
                "trade_count": trade_count,
                "sharpe": sharpe,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "max_dd": max_dd,
                "capital_used": capital_used,
                "dashboard_ret_n": np.nan,
                "dashboard_annual_ret": np.nan,
                "dashboard_win_rate": np.nan,
                "dashboard_drawdown_dollar": np.nan,
                "dashboard_pnl_n": np.nan,
                "monotonicity_flag": False,
                "passes_min_trades": trade_count >= static_config.min_trades,
                "passes_positive_sharpe": bool(pd.notna(sharpe) and sharpe > 0),
                "passes_monotonicity": False,
                "validation_pass": False,
                "crit_list_json": crit_list_to_literal(crit_list),
            },
            "dashboard_df": pd.DataFrame(),
            "dashboard_bydate_df": pd.DataFrame(),
            "optmz_summary_df": optmz_summary if isinstance(optmz_summary, pd.DataFrame) else pd.DataFrame(),
            "rule_metrics": shared_metrics,
        }

    strat_df, _, dashboard_bydate_df = dashboard(
        optmz_events,
        static_config.entry_fee,
        f"{method_name}_{candidate_name}_{PARTITION_OUTPUT_LABELS[partition_name]}",
        static_config.long_short,
        "complete",
        "max",
    )

    dashboard_metrics = _extract_dashboard_metrics(strat_df)
    monotonicity_flag = _is_monotonic_equity_curve(dashboard_bydate_df, "cum_pl_n")

    return {
        "summary_row": {
            "method_name": method_name,
            "candidate_name": candidate_name,
            "partition_name": partition_name,
            "partition_label": PARTITION_OUTPUT_LABELS[partition_name],
            "trade_count": trade_count,
            "sharpe": sharpe,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "max_dd": max_dd,
            "capital_used": capital_used,
            "dashboard_ret_n": dashboard_metrics.get("ret_n", np.nan),
            "dashboard_annual_ret": dashboard_metrics.get("annual ret", np.nan),
            "dashboard_win_rate": dashboard_metrics.get("daily win", np.nan),
            "dashboard_drawdown_dollar": dashboard_metrics.get("max drawday dollar", np.nan),
            "dashboard_pnl_n": dashboard_metrics.get("pl_n", np.nan),
            "monotonicity_flag": monotonicity_flag,
            "passes_min_trades": bool(trade_count >= static_config.min_trades),
            "passes_positive_sharpe": bool(pd.notna(sharpe) and sharpe > 0),
            "passes_monotonicity": monotonicity_flag,
            "validation_pass": bool(
                trade_count >= static_config.min_trades and pd.notna(sharpe) and sharpe > 0 and monotonicity_flag
            ),
            "crit_list_json": crit_list_to_literal(crit_list),
        },
        "dashboard_df": strat_df,
        "dashboard_bydate_df": dashboard_bydate_df,
        "optmz_summary_df": optmz_summary,
        "rule_metrics": shared_metrics,
    }


def build_search_record(
    method_name: str,
    candidate_name: str,
    feature_pool: List[str],
    search_result: Any,
    raw_validation: Dict[str, Any],
    rounded_validation: Dict[str, Any],
) -> Dict[str, Any]:
    rounded_summary = rounded_validation["summary_row"]
    raw_summary = raw_validation["summary_row"]
    return {
        "method_name": method_name,
        "candidate_name": candidate_name,
        "feature_pool_count": len(feature_pool),
        "selected_feature_count": len(search_result.chosen),
        "crit_list_json": rounded_summary["crit_list_json"],
        "search_trade_count": search_result.trade_count,
        "search_sharpe": search_result.sharpe,
        "search_ann_return": search_result.ann_return,
        "search_ann_vol": search_result.ann_vol,
        "search_max_dd": search_result.max_dd,
        "search_capital_used": search_result.capital_used,
        "reapplied_trade_count_ins_80": raw_summary["trade_count"],
        "reapplied_sharpe_ins_80": raw_summary["sharpe"],
        "reapplied_capital_used_ins_80": raw_summary["capital_used"],
        "search_apply_trade_delta_ins_80": raw_summary["trade_count"] - search_result.trade_count,
        "search_apply_sharpe_delta_ins_80": raw_summary["sharpe"] - search_result.sharpe,
        "rounded_trade_count_ins_80": rounded_summary["trade_count"],
        "rounded_sharpe_ins_80": rounded_summary["sharpe"],
        "rounded_ann_return_ins_80": rounded_summary["ann_return"],
        "rounded_ann_vol_ins_80": rounded_summary["ann_vol"],
        "rounded_max_dd_ins_80": rounded_summary["max_dd"],
        "rounded_capital_used_ins_80": rounded_summary["capital_used"],
        "monotonicity_flag_ins_80": rounded_summary["monotonicity_flag"],
        "passes_min_trades_ins_80": rounded_summary["passes_min_trades"],
        "passes_positive_sharpe_ins_80": rounded_summary["passes_positive_sharpe"],
        "passes_monotonicity_ins_80": rounded_summary["passes_monotonicity"],
        "validation_pass_ins_80": rounded_summary["validation_pass"],
        "max_capital_global": search_result.logs.get("max_capital_global"),
    }


def _cleanup_best_json_if_disabled(path: Path, save_best_json: bool) -> None:
    if save_best_json or not path.exists():
        return
    try:
        path.unlink()
    except OSError as exc:
        warnings.warn(f"Unable to remove stale best-json artifact: {path} ({exc})")


def export_search_artifacts(
    method_name: str,
    candidate_df: pd.DataFrame,
    csv_dir: Path,
    json_dir: Path,
    save_best_json: bool,
) -> Tuple[pd.DataFrame, Path, Optional[Path]]:
    ranked_df = candidate_df.sort_values(
        by=[
            "validation_pass_ins_80",
            "passes_positive_sharpe_ins_80",
            "passes_monotonicity_ins_80",
            "passes_min_trades_ins_80",
            "rounded_sharpe_ins_80",
            "search_sharpe",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    csv_path = _save_dataframe_csv(ranked_df, csv_dir / f"{method_name}_candidates.csv")
    best_json_path = json_dir / f"{method_name}_best.json"
    json_path: Optional[Path] = None
    if save_best_json and not ranked_df.empty:
        json_path = _save_json(ranked_df.iloc[0].to_dict(), best_json_path)
    else:
        _cleanup_best_json_if_disabled(best_json_path, save_best_json=save_best_json)
    return ranked_df, csv_path, json_path


def build_feature_set_registry(partitions: Dict[str, pd.DataFrame], probability_column: str) -> Dict[str, List[str]]:
    partition_ins_80 = partitions["partition_ins_80_001"]
    partition_ins_20 = partitions["partition_ins_20_001"]
    partition_oos = partitions["partition_oos_001"]

    ins80_feature_columns = build_feature_list_from_columns(partition_ins_80)
    common_feature_columns = [
        col
        for col in ins80_feature_columns
        if col in partition_ins_20.columns and col in partition_oos.columns
    ]
    if not common_feature_columns:
        raise ValueError("No shared feature columns were selected from the current partitions.")

    require_columns(partition_ins_80, common_feature_columns, "partition_ins_80_001 feature universe")
    require_columns(partition_ins_20, common_feature_columns, "partition_ins_20_001 feature universe")
    require_columns(partition_oos, common_feature_columns, "partition_oos_001 feature universe")

    probability_features = (
        [probability_column]
        if all(probability_column in df.columns for df in (partition_ins_80, partition_ins_20, partition_oos))
        else []
    )
    feats_a = _safe_feature_subset(common_feature_columns, GROUP_ALWAYS)
    feats_b = [col for col in common_feature_columns if col.startswith("dist_")]
    feats_c = [col for col in common_feature_columns if col.startswith("pct_")]
    feats_d = [col for col in common_feature_columns if col.startswith("ret_")]

    registry = {
        "all_plus_ml": list(dict.fromkeys(common_feature_columns + probability_features)),
        "no_ml": common_feature_columns,
        "branch_a": feats_a,
        "branch_b": feats_b,
        "branch_c": feats_c,
        "branch_d": feats_d,
        "branch_a_ml": list(dict.fromkeys(feats_a + probability_features)),
        "branch_b_ml": list(dict.fromkeys(feats_b + probability_features)),
        "branch_c_ml": list(dict.fromkeys(feats_c + probability_features)),
        "branch_d_ml": list(dict.fromkeys(feats_d + probability_features)),
    }
    return {name: cols for name, cols in registry.items() if cols}


def _run_greedy_search(
    partitions: Dict[str, pd.DataFrame],
    feature_set_registry: Dict[str, List[str]],
    static_config: StaticOptimizationConfig,
    csv_dir: Path,
    json_dir: Path,
) -> SearchMethodArtifacts:
    partition_ins_80 = partitions["partition_ins_80_001"]
    search_records: List[Dict[str, Any]] = []

    for candidate_name, candidate_features in feature_set_registry.items():
        res = greedy_threshold_search(
            partition_ins_80,
            candidate_features,
            n_quantiles=static_config.greedy_n_quantiles,
            max_features=static_config.greedy_max_features,
            improvement_eps=static_config.greedy_improvement_eps,
            adaptive_tails=True,
            min_count_per_side=static_config.search_min_count_per_side,
            crit_round_digits=static_config.round_crit_digits,
        )
        raw_validation = summarize_partition_rule(
            partition_ins_80,
            chosen_to_crit_list(res.chosen),
            "partition_ins_80_001",
            "greedy_threshold_search",
            candidate_name,
            static_config=static_config,
        )
        rounded_validation = summarize_partition_rule(
            partition_ins_80,
            round_crit_list(res.crit_list, overrides=static_config.feature_threshold_overrides),
            "partition_ins_80_001",
            "greedy_threshold_search",
            candidate_name,
            static_config=static_config,
        )
        search_records.append(
            build_search_record(
                "greedy_threshold_search",
                candidate_name,
                candidate_features,
                res,
                raw_validation,
                rounded_validation,
            )
        )

    candidate_df = pd.DataFrame(search_records)
    ranked_df, csv_path, json_path = export_search_artifacts(
        "greedy_threshold_search",
        candidate_df,
        csv_dir=csv_dir,
        json_dir=json_dir,
        save_best_json=static_config.save_best_json,
    )
    return SearchMethodArtifacts(
        method_name="greedy_threshold_search",
        ranked_candidates=ranked_df,
        candidate_csv_path=csv_path,
        best_json_path=json_path,
        selected_row=ranked_df.iloc[0].to_dict() if not ranked_df.empty else {},
    )


def _run_bayesian_search(
    partitions: Dict[str, pd.DataFrame],
    feature_set_registry: Dict[str, List[str]],
    static_config: StaticOptimizationConfig,
    csv_dir: Path,
    json_dir: Path,
) -> SearchMethodArtifacts:
    partition_ins_80 = partitions["partition_ins_80_001"]
    bayesian_feature_sets = {
        key: value
        for key, value in feature_set_registry.items()
        if key not in {"all_plus_ml", "no_ml"}
    }
    search_records: List[Dict[str, Any]] = []

    for candidate_name, candidate_features in bayesian_feature_sets.items():
        res = bayesian_greedy_threshold_search(
            df=partition_ins_80,
            features=candidate_features,
            mtm_col="mtm_pl",
            date_col="normed_date",
            max_features=static_config.greedy_max_features,
            improvement_eps=static_config.greedy_improvement_eps,
            seed=123,
            use_numexpr=True,
            n_calls_per_step=static_config.bayesian_n_calls_per_step,
            n_random_starts=static_config.bayesian_n_random_starts,
            tail_penalty=static_config.bayesian_tail_penalty,
            complexity_penalty=static_config.bayesian_complexity_penalty,
            lock_to_grid=static_config.bayesian_lock_to_grid,
            n_quantiles_lock=static_config.bayesian_n_quantiles_lock,
            crit_round_digits=static_config.round_crit_digits,
        )
        raw_validation = summarize_partition_rule(
            partition_ins_80,
            chosen_to_crit_list(res.chosen),
            "partition_ins_80_001",
            "bayesian_greedy_threshold_search",
            candidate_name,
            static_config=static_config,
        )
        rounded_validation = summarize_partition_rule(
            partition_ins_80,
            round_crit_list(res.crit_list, overrides=static_config.feature_threshold_overrides),
            "partition_ins_80_001",
            "bayesian_greedy_threshold_search",
            candidate_name,
            static_config=static_config,
        )
        search_records.append(
            build_search_record(
                "bayesian_greedy_threshold_search",
                candidate_name,
                candidate_features,
                res,
                raw_validation,
                rounded_validation,
            )
        )

    candidate_df = pd.DataFrame(search_records)
    ranked_df, csv_path, json_path = export_search_artifacts(
        "bayesian_greedy_threshold_search",
        candidate_df,
        csv_dir=csv_dir,
        json_dir=json_dir,
        save_best_json=static_config.save_best_json,
    )
    return SearchMethodArtifacts(
        method_name="bayesian_greedy_threshold_search",
        ranked_candidates=ranked_df,
        candidate_csv_path=csv_path,
        best_json_path=json_path,
        selected_row=ranked_df.iloc[0].to_dict() if not ranked_df.empty else {},
    )


def _build_selected_rule_reviews(
    static_config: StaticOptimizationConfig,
    partitions: Dict[str, pd.DataFrame],
    method_results: Dict[str, SearchMethodArtifacts],
    csv_dir: Path,
) -> Tuple[Dict[str, SelectedRuleReview], pd.DataFrame, Path]:
    selected_frames = {
        "greedy_best": method_results["greedy_threshold_search"].ranked_candidates,
        "bayesian_best": method_results["bayesian_greedy_threshold_search"].ranked_candidates,
    }

    summary_rows: List[Dict[str, Any]] = []
    reviews: Dict[str, SelectedRuleReview] = {}

    for rule_name, ranked_df in selected_frames.items():
        if ranked_df.empty:
            continue

        top_row = ranked_df.iloc[0].to_dict()
        method_name = str(top_row["method_name"])
        candidate_name = str(top_row["candidate_name"])
        crit_list = normalize_crit_list(top_row["crit_list_json"])

        partition_dashboard_frames: List[pd.DataFrame] = []
        partition_curve_frames: List[pd.DataFrame] = []
        bydate_map: Dict[str, pd.DataFrame] = {}

        for partition_name, partition_df in partitions.items():
            evaluation = summarize_partition_rule(
                partition_df,
                crit_list,
                partition_name,
                method_name,
                candidate_name,
                static_config=static_config,
            )
            summary_row = dict(evaluation["summary_row"])
            summary_row["rule_name"] = rule_name
            summary_rows.append(summary_row)

            if not evaluation["dashboard_df"].empty:
                dashboard_export_df = evaluation["dashboard_df"].copy()
                dashboard_export_df["partition_name"] = partition_name
                dashboard_export_df["rule_name"] = rule_name
                dashboard_export_df["candidate_name"] = candidate_name
                partition_dashboard_frames.append(dashboard_export_df)
                if static_config.save_partition_dashboards_debug:
                    _save_dataframe_csv(
                        dashboard_export_df,
                        csv_dir / f"dashboard_{rule_name}_{PARTITION_OUTPUT_LABELS[partition_name]}.csv",
                    )

            if not evaluation["dashboard_bydate_df"].empty:
                bydate_export_df = evaluation["dashboard_bydate_df"].copy()
                bydate_export_df["partition_name"] = partition_name
                bydate_export_df["rule_name"] = rule_name
                bydate_export_df["candidate_name"] = candidate_name
                partition_curve_frames.append(bydate_export_df)
                bydate_map[partition_name] = evaluation["dashboard_bydate_df"].copy()
                if static_config.save_partition_dashboards_debug and static_config.save_equity_curve:
                    _save_dataframe_csv(
                        bydate_export_df,
                        csv_dir / f"dashboard_{rule_name}_{PARTITION_OUTPUT_LABELS[partition_name]}_equity_curve.csv",
                    )

        combined_dashboard_df = (
            pd.concat(partition_dashboard_frames, ignore_index=True)
            if partition_dashboard_frames
            else pd.DataFrame()
        )
        combined_dashboard_path = (
            _save_dataframe_csv(combined_dashboard_df, csv_dir / f"dashboard_{rule_name}_all_partitions.csv")
            if not combined_dashboard_df.empty
            else None
        )

        combined_curve_df = (
            pd.concat(partition_curve_frames, ignore_index=True)
            if partition_curve_frames
            else pd.DataFrame()
        )
        combined_curve_path = (
            _save_dataframe_csv(combined_curve_df, csv_dir / f"dashboard_{rule_name}_equity_curves.csv")
            if static_config.save_equity_curve and not combined_curve_df.empty
            else None
        )

        reviews[rule_name] = SelectedRuleReview(
            rule_name=rule_name,
            method_name=method_name,
            candidate_name=candidate_name,
            crit_list_json=str(top_row["crit_list_json"]),
            combined_dashboard_df=combined_dashboard_df,
            combined_dashboard_path=combined_dashboard_path,
            combined_curve_df=combined_curve_df,
            combined_curve_path=combined_curve_path,
            bydate_map=bydate_map,
        )

    validation_summary = pd.DataFrame(summary_rows)
    if not validation_summary.empty:
        group_cols = ["method_name", "candidate_name", "rule_name"]
        validation_summary["cross_partition_positive_sharpe"] = validation_summary.groupby(group_cols)["passes_positive_sharpe"].transform("all")
        validation_summary["cross_partition_min_trades"] = validation_summary.groupby(group_cols)["passes_min_trades"].transform("all")
        validation_summary["cross_partition_monotonicity"] = validation_summary.groupby(group_cols)["passes_monotonicity"].transform("all")
        validation_summary["cross_partition_validation_pass"] = validation_summary.groupby(group_cols)["validation_pass"].transform("all")

    validation_summary_path = _save_dataframe_csv(validation_summary, csv_dir / "threshold_validation_summary.csv")
    return reviews, validation_summary, validation_summary_path


def run_static_optimization_stage(
    run_config: RunConfig,
    static_config: Optional[StaticOptimizationConfig] = None,
    training: Optional[TrainStageResult] = None,
) -> StaticOptimizationStageResult:
    static_config = static_config or StaticOptimizationConfig(
        long_short=run_config.long_short,
        entry_fee=run_config.entry_fee,
    )
    output_dirs = _static_output_dirs(run_config, static_config)
    partitions = _resolve_scored_partitions(run_config, training=training)
    feature_set_registry = build_feature_set_registry(partitions, probability_column=run_config.probability_column)

    method_results = {
        "greedy_threshold_search": _run_greedy_search(
            partitions,
            feature_set_registry,
            static_config=static_config,
            csv_dir=output_dirs["csv_dir"],
            json_dir=output_dirs["json_dir"],
        ),
        "bayesian_greedy_threshold_search": _run_bayesian_search(
            partitions,
            feature_set_registry,
            static_config=static_config,
            csv_dir=output_dirs["csv_dir"],
            json_dir=output_dirs["json_dir"],
        ),
    }

    selected_rule_reviews, validation_summary, validation_summary_path = _build_selected_rule_reviews(
        static_config=static_config,
        partitions=partitions,
        method_results=method_results,
        csv_dir=output_dirs["csv_dir"],
    )

    return StaticOptimizationStageResult(
        run_config=run_config,
        static_config=static_config,
        partitions=partitions,
        method_results=method_results,
        selected_rule_reviews=selected_rule_reviews,
        validation_summary=validation_summary,
        validation_summary_path=validation_summary_path,
    )


__all__ = [
    "SearchMethodArtifacts",
    "SelectedRuleReview",
    "StaticOptimizationConfig",
    "StaticOptimizationStageResult",
    "build_feature_set_registry",
    "chosen_to_crit_list",
    "crit_list_to_literal",
    "crit_list_to_rule_dict",
    "evaluate_rule_metrics",
    "normalize_crit_list",
    "round_crit_list",
    "round_threshold_for_feature",
    "run_static_optimization_stage",
    "summarize_partition_rule",
]
