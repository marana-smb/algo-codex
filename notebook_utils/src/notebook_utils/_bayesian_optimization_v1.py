from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._fast_optimization_v1 import (
    HAS_NUMEXPR,
    Candidate,
    SearchResult,
    _adaptive_bounds,
    _build_mask_numexpr,
    _build_mask_numpy,
    _estimate_capital_proxy_rows,
    _estimate_max_capital_global,
    _evaluate_mask_metrics,
    _max_drawdown,
    _max_drawdown_with_indices,
    _quantile_thresholds,
    _to_crit_list,
    _to_day_ids,
    greedy_threshold_search,
)

"""
This function, bayesian_greedy_threshold_search, implements an iterative, 
greedy optimization algorithm to find a set of simple filter conditions 
(predicates) that maximize a financial metric, typically the Sharpe Ratio, 
for a given set of trading results.

It uses Bayesian Optimization (BO) via skopt.gp_minimize to intelligently 
search for the best combination of feature, comparison operator, and 
threshold in each step. The process is "greedy" because it finds the 
single best feature/threshold/operator at each step and adds it to the filter set, then repeats.

Bayesian greedy threshold search (no CV) with:
- Exposed BO knobs (n_calls_per_step, tail_penalty, complexity_penalty).
- Optional reproducible threshold locking to a quantile grid.

Requires your existing helpers in scope:
    Candidate, SearchResult, _to_day_ids, _annualized_sharpe, _max_drawdown,
    _aggregate_daily_sum, _estimate_max_capital_global,
    _build_mask_numpy, _build_mask_numexpr, HAS_NUMEXPR,
    _adaptive_bounds, _quantile_thresholds, greedy_threshold_search
"""

# --- utils ---
def _finite_quantile(arr: np.ndarray, q: float) -> float:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return float("nan")
    try:
        return float(np.quantile(a, q, method="linear"))
    except TypeError:
        return float(np.quantile(a, q, interpolation="linear"))

def _eval_one_candidate(
    daily_pl: np.ndarray,
    cap_row: np.ndarray,
    day_id: np.ndarray,
    cols: Dict[str, np.ndarray],
    chosen: List[Candidate],
    cand: Candidate,
    n_days: int,
    use_numexpr: bool,
) -> Dict[str, Any]:
    build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
    mask = build(cols, chosen + [cand])
    return _evaluate_mask_metrics(daily_pl, cap_row, day_id, mask, n_days)

def _lock_threshold_quantile_grid(x_surv: np.ndarray, q_low: float, q_high: float, n_quantiles_lock: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (thrs, qs) grid for reproducible snapping."""
    thrs = _quantile_thresholds(x_surv, n_quantiles_lock, q_low=q_low, q_high=q_high)
    if thrs.size == 0:
        return thrs, np.array([], dtype=np.float32)
    qs = np.linspace(q_low, q_high, num=thrs.size, dtype=np.float32)
    return thrs, qs

def bayesian_greedy_threshold_search(
    df: pd.DataFrame,
    features: List[str],
    *,
    mtm_col: str = "mtm_pl",
    date_col: str = "normed_date",
    max_features: int = 6,
    improvement_eps: float = 0.02,
    seed: int = 42,
    use_numexpr: bool = True,
    # BO knobs
    n_calls_per_step: int = 30,
    n_random_starts: int = 8,
    adaptive_tails: bool = True,
    base_q_low: float = 0.05,
    base_q_high: float = 0.95,
    min_count_per_side: int = 200,
    tail_penalty: float = 0.5,        # discourage extreme quantiles
    complexity_penalty: float = 0.02, # penalize each predicate added
    # Reproducible grid lock
    lock_to_grid: bool = False,
    n_quantiles_lock: int = 24,       # grid size if lock_to_grid=True
    crit_round_digits: int = 4,
) -> SearchResult:
    """
    BO selects (feature, op, quantile) using overall Sharpe.
    Objective = (Sharpe after − baseline) − tail_penalty − complexity_penalty * (k+1).
    If lock_to_grid=True, snap threshold to nearest quantile grid for reproducibility.
    """
    rng = np.random.default_rng(seed)

    pl_rows = df[mtm_col].astype(np.float64).to_numpy(copy=False)
    day_id, uniq_days = _to_day_ids(df, date_col=date_col)
    n_days = int(uniq_days.size)
    cap_row = _estimate_capital_proxy_rows(df, mtm_col=mtm_col)
    cols: Dict[str, np.ndarray] = {f: df[f].astype(np.float32).to_numpy(copy=False) for f in features}

    max_capital_global = _estimate_max_capital_global(df, day_id, n_days, mtm_col=mtm_col)
    baseline_metrics = _evaluate_mask_metrics(pl_rows, cap_row, day_id, np.ones(len(df), dtype=bool), n_days)
    base_sharpe = float(baseline_metrics["sharpe"])

    chosen: List[Candidate] = []
    best_metrics = baseline_metrics
    best_daily: Optional[np.ndarray] = None
    remaining = set(features)

    # Try scikit-optimize; fallback if missing
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Real
    except Exception:
        return greedy_threshold_search(
            df, features,
            mtm_col=mtm_col, date_col=date_col, n_quantiles=24,
            max_features=max_features, improvement_eps=improvement_eps,
            seed=seed, use_numexpr=use_numexpr, random_subset_per_feat=48,
            crit_round_digits=4, adaptive_tails=adaptive_tails,
            base_q_low=base_q_low, base_q_high=base_q_high,
            min_count_per_side=min_count_per_side
        )

    bo_logs: List[Dict[str, Any]] = []

    for step in range(max_features):
        build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
        base_mask = build(cols, chosen) if chosen else np.ones_like(day_id, dtype=bool)

        # Per-feature q-bounds
        q_bounds: Dict[str, Tuple[float, float]] = {}
        viable_feats: List[str] = []
        for f in list(remaining):
            x_surv = cols[f][base_mask]
            n_sub = int(np.isfinite(x_surv).sum())
            if n_sub <= 0:
                continue
            if adaptive_tails:
                low, high = _adaptive_bounds(n_sub, base_q_low, base_q_high, min_count_per_side)
            else:
                low, high = base_q_low, base_q_high
            if low >= high:
                continue
            q_bounds[f] = (float(low), float(high))
            viable_feats.append(f)

        if not viable_feats:
            break

        baseline = float(best_metrics["sharpe"]) if chosen else base_sharpe
        space = [
            Categorical(viable_feats, name="feat"),
            Categorical([">=", "<="], name="op"),
            Real(0.0, 1.0, name="q_proxy"),
        ]

        def objective(params: List[Any]) -> float:
            feat, op, q_proxy = params
            if feat not in q_bounds:
                return 0.0
            ql, qh = q_bounds[feat]
            q = ql + np.clip(q_proxy, 0.0, 1.0) * (qh - ql)

            x_surv = cols[feat][base_mask]
            thr = _finite_quantile(x_surv, float(q))
            if not np.isfinite(thr):
                return 0.0

            if lock_to_grid and n_quantiles_lock > 0:
                thrs_grid, _ = _lock_threshold_quantile_grid(x_surv, ql, qh, n_quantiles_lock)
                if thrs_grid.size:
                    j = int(np.argmin(np.abs(thrs_grid - thr)))
                    thr = float(thrs_grid[j])

            cand = Candidate(feat=feat, threshold=float(thr), op=str(op))
            metrics = _eval_one_candidate(pl_rows, cap_row, day_id, cols, chosen, cand, n_days, use_numexpr)
            s = float(metrics["sharpe"])
            improvement = s - baseline

            mid = 0.5 * (ql + qh)
            tail_dist = abs(float(q) - mid) / max(1e-6, (qh - ql) / 2.0)
            pen_tail = tail_penalty * tail_dist
            pen_complex = complexity_penalty * (len(chosen) + 1)

            score = improvement - pen_tail - pen_complex
            return -float(score)

        res = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=int(n_calls_per_step),
            n_random_starts=int(n_random_starts),
            acq_func="EI",
            random_state=int(seed + step),
            noise="gaussian",
        )

        best_feat, best_op, best_q_proxy = res.x
        ql, qh = q_bounds[best_feat]
        best_q = ql + np.clip(best_q_proxy, 0.0, 1.0) * (qh - ql)

        x_surv = cols[best_feat][base_mask]
        thr = _finite_quantile(x_surv, float(best_q))
        if not np.isfinite(thr):
            break

        if lock_to_grid and n_quantiles_lock > 0:
            thrs_grid, qs_grid = _lock_threshold_quantile_grid(x_surv, ql, qh, n_quantiles_lock)
            if thrs_grid.size:
                j = int(np.argmin(np.abs(thrs_grid - thr)))
                thr = float(thrs_grid[j])
                best_q = float(qs_grid[j])

        cand = Candidate(best_feat, float(thr), str(best_op))
        cand_metrics = _eval_one_candidate(pl_rows, cap_row, day_id, cols, chosen, cand, n_days, use_numexpr)
        cand_s = float(cand_metrics["sharpe"])

        net_improvement = (cand_s - baseline) - (complexity_penalty * (len(chosen) + 1))
        if net_improvement < improvement_eps:
            break

        chosen.append(cand)
        if cand.feat in remaining:
            remaining.remove(cand.feat)
        best_metrics = cand_metrics
        best_daily = np.asarray(cand_metrics["daily_ret"], dtype=np.float64)

        bo_logs.append({
            "step": step,
            "chosen_feature": best_feat,
            "op": best_op,
            "q": float(best_q),
            "threshold": float(thr),
            "sharpe_after": float(cand_s),
            "net_improvement": float(net_improvement),
            "evaluations": int(len(res.func_vals)),
            "best_objective": float(res.fun),
            "lock_to_grid": bool(lock_to_grid),
            "n_quantiles_lock": int(n_quantiles_lock),
        })

    if best_daily is None:
        best_daily = np.asarray(baseline_metrics["daily_ret"], dtype=np.float64)
        best_metrics = baseline_metrics
    max_dd_final, peak_idx, trough_idx = _max_drawdown_with_indices(best_daily)
    best_metrics["max_dd"] = max_dd_final

    crit_list = _to_crit_list(chosen, ndigits=crit_round_digits, combined=True)
    logs = {
        "max_capital_global": max_capital_global,
        "capital_used": float(best_metrics["capital_used"]),
        "trade_count": int(best_metrics["trade_count"]),
        "bo_trace": bo_logs,
        "drawdown": {
            "max_dd": float(best_metrics["max_dd"]),
            "peak_idx": peak_idx,
            "trough_idx": trough_idx,
        },
        "settings": {
            "bayes": True,
            "n_calls_per_step": n_calls_per_step,
            "n_random_starts": n_random_starts,
            "adaptive_tails": adaptive_tails,
            "base_q_low": base_q_low,
            "base_q_high": base_q_high,
            "min_count_per_side": min_count_per_side,
            "tail_penalty": tail_penalty,
            "complexity_penalty": complexity_penalty,
            "lock_to_grid": lock_to_grid,
            "n_quantiles_lock": n_quantiles_lock,
            "seed": seed,
        },
    }
    return SearchResult(
        sharpe=float(best_metrics["sharpe"]),
        ann_return=float(best_metrics["ann_return"]),
        ann_vol=float(best_metrics["ann_vol"]),
        max_dd=float(best_metrics["max_dd"]),
        daily_series=best_daily,
        chosen=chosen,
        crit_list=crit_list,
        trade_count=int(best_metrics["trade_count"]),
        capital_used=float(best_metrics["capital_used"]),
        logs=logs,
    )

# --------------------------- Usage example ---------------------------
# Install once in notebook (Py 3.7):
# import sys; !{sys.executable} -m pip install "scikit-optimize==0.9.0" "scikit-learn<=1.0.2"
#
# Example call:
# res = bayesian_greedy_threshold_search(
#     df=train_df,
#     features=["f1","f2","f3","f4"],
#     mtm_col="mtm_pl",
#     date_col="normed_date",
#     max_features=5,
#     improvement_eps=0.015,
#     seed=123,
#     use_numexpr=True,
#     # BO knobs
#     n_calls_per_step=35,
#     n_random_starts=10,
#     tail_penalty=0.4,
#     complexity_penalty=0.03,
#     # Reproducible locking
#     lock_to_grid=True,
#     n_quantiles_lock=32,
# )
# print("Sharpe:", res.sharpe, "Chosen:", res.chosen)
# print("Settings:", res.logs["settings"])
# for step in res.logs["bo_trace"]:
#     print(step)
