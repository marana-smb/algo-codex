
from __future__ import annotations
import warnings, importlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import _fast_optimization_v1

from _fast_optimization_v1 import greedy_threshold_search

# ---------------- Data structures ----------------
@dataclass(frozen=True)
class Candidate:
    feat: str
    threshold: float
    op: str  # ">=" or "<="

@dataclass
class SearchResult:
    sharpe: float
    ann_return: float
    ann_vol: float
    max_dd: float
    daily_series: np.ndarray             # daily returns (not P&L)
    chosen: List[Candidate]
    crit_list: List[Dict[str, Tuple[float, str]]]
    logs: Dict[str, object]              # {'max_capital_global': float, 'trials_per_feature': list, ...}

# ---------------- Core helpers ----------------
def _to_day_ids(df: pd.DataFrame, date_col: str = "normed_date") -> Tuple[np.ndarray, np.ndarray]:
    dts = pd.to_datetime(df[date_col], errors="coerce")
    try:
        ns = dts.view("int64")
    except Exception:
        ns = dts.astype("int64")
    days = ns // 86_400_000_000_000  # ns→days
    uniq, day_id = np.unique(days, return_inverse=True)
    return day_id.astype(np.int32, copy=False), uniq

def _annualized_sharpe(daily_returns: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    x = daily_returns.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mu = x.mean()
    sigma = x.std(ddof=1)
    ann_ret = 252 * mu
    ann_vol = np.sqrt(252) * sigma
    return (ann_ret / (ann_vol + eps), ann_ret, ann_vol)

# def _max_drawdown(daily_returns: np.ndarray) -> float:
#     eq = np.cumsum(daily_returns)
#     peak = np.maximum.accumulate(eq)
#     return float((eq - peak).min())

def _max_drawdown(daily_returns: np.ndarray) -> float:
    x = daily_returns.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())

def _max_drawdown_with_indices(daily_returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Max percentage drawdown on compounded equity.
    Returns:
        max_dd   : negative drawdown value in [-1, 0]
        peak_idx : index of peak before the worst trough (0-based)
        trough_idx: index of worst trough (0-based)
    """
    x = daily_returns.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, -1, -1
    eq = np.cumprod(1.0 + x)                  # compounded equity
    peak = np.maximum.accumulate(eq)          # running HWM
    dd = eq / peak - 1.0                      # in [-1, 0]
    trough = int(np.argmin(dd))
    # peak index up to trough (first occurrence of the HWM works)
    peak_idx = int(np.argmax(eq[:trough + 1]))
    return float(dd.min()), peak_idx, trough

def _aggregate_daily_sum(values: np.ndarray, day_id: np.ndarray, mask: Optional[np.ndarray], n_days: int) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(day_id, dtype=bool)
    good = mask & np.isfinite(values)
    return np.bincount(day_id[good], weights=values[good], minlength=n_days).astype(np.float64, copy=False)

def _estimate_max_capital_global(
    df: pd.DataFrame,
    day_id: np.ndarray,
    n_days: int,
    *,
    entry_shares_col: str = "entry_shares",
    entry_price_col: str = "entry_price",
    matched_shares_col: str = "matched_shares",
    exit_shares_col: str = "exit_shares",
    exit_price_col: str = "exit_price",
) -> float:
    """
    Estimate a scalar denominator: MAX over days of the SUM capital used that day.
    Capital proxy priority:
      1) abs(entry_shares * entry_price)
      2) abs(matched_shares * entry_price)
      3) abs(exit_shares * exit_price)
    """
    cap_row: Optional[np.ndarray] = None
    if entry_shares_col in df and entry_price_col in df:
        cap_row = np.abs(df[entry_shares_col].to_numpy(dtype=np.float64, copy=False)
                         * df[entry_price_col].to_numpy(dtype=np.float64, copy=False))
    elif matched_shares_col in df and entry_price_col in df:
        cap_row = np.abs(df[matched_shares_col].to_numpy(dtype=np.float64, copy=False)
                         * df[entry_price_col].to_numpy(dtype=np.float64, copy=False))
    elif exit_shares_col in df and exit_price_col in df:
        cap_row = np.abs(df[exit_shares_col].to_numpy(dtype=np.float64, copy=False)
                         * df[exit_price_col].to_numpy(dtype=np.float64, copy=False))
    else:
        warnings.warn("Could not infer capital proxy; falling back to |mtm_pl| for scale (rough).")
        cap_row = np.abs(df["mtm_pl"].to_numpy(dtype=np.float64, copy=False))

    cap_daily = _aggregate_daily_sum(cap_row, day_id, mask=None, n_days=n_days)
    max_cap = float(np.nanmax(cap_daily)) if cap_daily.size else 0.0
    if not np.isfinite(max_cap) or max_cap <= 0.0:
        warnings.warn("Estimated max capital <= 0; Sharpe on returns may be ill-defined. Using 1.0.")
        max_cap = 1.0
    return max_cap

# ---------------- Masks ----------------
def _build_mask_numpy(cols: Dict[str, np.ndarray], chosen: List[Candidate]) -> np.ndarray:
    if not chosen:
        n = next(iter(cols.values())).shape[0]
        return np.ones(n, dtype=bool)
    n = next(iter(cols.values())).shape[0]
    mask = np.ones(n, dtype=bool)
    for c in chosen:
        x = cols[c.feat]
        thr = np.asarray(c.threshold, dtype=x.dtype)
        pred = (x >= thr) if c.op == ">=" else (x <= thr)
        pred &= np.isfinite(x)
        mask &= pred
        if not mask.any():
            break
    return mask

try:
    import numexpr as ne
    HAS_NUMEXPR = True
except Exception:
    HAS_NUMEXPR = False

def _build_mask_numexpr(cols: Dict[str, np.ndarray], chosen: List[Candidate]) -> np.ndarray:
    """
    Patched: Avoid numexpr function calls (e.g., isfinite()) for broad compatibility.
    We precompute a finite-mask per column and AND it in via a variable.
    """
    if not chosen:
        n = next(iter(cols.values())).shape[0]
        return np.ones(n, dtype=bool)

    expr_parts: List[str] = []
    local_dict: Dict[str, Union[np.ndarray, float, bool]] = {}

    for i, c in enumerate(chosen):
        xi, ti, fi = f"x{i}", f"t{i}", f"f{i}"
        x = cols[c.feat]
        local_dict[xi] = x
        local_dict[ti] = float(c.threshold)
        local_dict[fi] = np.isfinite(x)  # precomputed finite mask

        comp = f"({fi}) & ({xi}{c.op}{ti})"
        expr_parts.append(comp)

    expr = "&".join(expr_parts)
    mask = ne.evaluate(expr, local_dict=local_dict)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    return mask

# ---------------- Thresholds ----------------
def _quantile_thresholds(
    x: np.ndarray,
    n_q: int,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0 or n_q <= 0:
        return np.array([], dtype=np.float32)
    qs = np.linspace(q_low, q_high, n_q, dtype=np.float64)
    try:
        thr = np.quantile(x, qs, method="linear")
    except TypeError:
        thr = np.quantile(x, qs, interpolation="linear")
    return np.unique(thr.astype(np.float32))

def _adaptive_bounds(n_survive: int, base_low: float, base_high: float, min_count_per_side: int) -> Tuple[float, float]:
    if n_survive <= 0:
        return 0.25, 0.75  # degenerate fallback
    # shrink extremes so each tail has at least min_count_per_side rows
    frac = min(0.5, max(0.0, min_count_per_side / float(n_survive)))
    low = max(base_low, frac)
    high = min(base_high, 1.0 - frac)
    if low >= high:
        # extreme shrinkage fallback
        return 0.25, 0.75
    return low, high

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
    day_id: np.ndarray,
    cols: Dict[str, np.ndarray],
    chosen: List[Candidate],
    cand: Candidate,
    n_days: int,
    use_numexpr: bool,
    denominator: float,
) -> Tuple[float, float, float, float]:
    build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
    mask = build(cols, chosen + [cand])
    if not mask.any():
        return -np.inf, 0.0, 0.0, 0.0
    daily_sum = _aggregate_daily_sum(daily_pl, day_id, mask, n_days)
    daily_ret = daily_sum / denominator
    s, ar, av = _annualized_sharpe(daily_ret)
    dd = _max_drawdown(daily_ret)
    return s, ar, av, dd

def _lock_threshold_quantile_grid(x_surv: np.ndarray, q_low: float, q_high: float, n_quantiles_lock: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (thrs, qs) grid for reproducible snapping."""
    thrs = _quantile_thresholds(x_surv, n_quantiles_lock, q_low=q_low, q_high=q_high)
    if thrs.size == 0:
        return thrs, np.array([], dtype=np.float32)
    qs = np.linspace(q_low, q_high, num=thrs.size, dtype=np.float32)
    return thrs, qs

def _to_crit_list(
    chosen: List[Candidate],
    ndigits: int = 4,
    combined: bool = False,
) -> List[Dict[str, Tuple[float, str]]]:
    """
    When combined=False (default): [{'f1': (thr, op)}, {'f2': (thr, op)}, ...]
    When combined=True:           [{'f1': (thr, op), 'f2': (thr, op), ...}]
    """
    if not chosen:
        return [dict()] if combined else []
    if combined:
        d = {c.feat: (round(float(c.threshold), ndigits), c.op) for c in chosen}
        return [d]
    return [{c.feat: (round(float(c.threshold), ndigits), c.op)} for c in chosen]

def mask_from_crit_list(df: pd.DataFrame, crit_list: List[Dict[str, Tuple[float, str]]]) -> np.ndarray:
    """
    Reconstruct a boolean mask from a crit_list.
    Supports both formats returned by _to_crit_list(..., combined=True/False).
    """
    if not crit_list:
        return np.ones(len(df), dtype=bool)

    # Normalize to a flat dict: {feat: (thr, op), ...}
    if len(crit_list) == 1 and len(crit_list[0]) >= 1:
        crit_dict = crit_list[0]  # combined format
    else:
        crit_dict = {}
        for d in crit_list:       # list of singletons -> merge
            crit_dict.update(d)

    mask = np.ones(len(df), dtype=bool)
    for feat, (thr, op) in crit_dict.items():
        col = df[feat].to_numpy()
        cond = np.isfinite(col) & ((col >= thr) if op == ">=" else (col <= thr))
        mask &= cond
        if not mask.any():
            break
    return mask

# --- main ---
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
    crit_round_digits: int = 4
) -> SearchResult:
    """
    BO selects (feature, op, quantile) using overall Sharpe.
    Objective = (Sharpe after − baseline) − tail_penalty − complexity_penalty * (k+1).
    If lock_to_grid=True, snap threshold to nearest quantile grid for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Materialize arrays
    pl_rows = df[mtm_col].astype(np.float64).to_numpy(copy=False)
    day_id, uniq_days = _to_day_ids(df, date_col=date_col)
    n_days = int(uniq_days.size)
    cols: Dict[str, np.ndarray] = {f: df[f].astype(np.float32).to_numpy(copy=False) for f in features}

    # Denominator and baseline Sharpe
    max_capital_global = _estimate_max_capital_global(df, day_id, n_days)
    daily_pl_all = _aggregate_daily_sum(pl_rows, day_id, mask=None, n_days=n_days)
    daily_ret_all = daily_pl_all / max_capital_global
    base_sharpe, _, _ = _annualized_sharpe(daily_ret_all)

    chosen: List[Candidate] = []
    best_sharpe = -np.inf
    best_ann_ret = best_ann_vol = best_max_dd = 0.0
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
        # Surviving rows with current predicates
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

        baseline = best_sharpe if chosen else base_sharpe

        # BO space
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

            # Snap to grid (reproducibility) for evaluation too, so BO "knows" the discretization
            if lock_to_grid and n_quantiles_lock > 0:
                thrs_grid, qs_grid = _lock_threshold_quantile_grid(x_surv, ql, qh, n_quantiles_lock)
                if thrs_grid.size:
                    j = int(np.argmin(np.abs(thrs_grid - thr)))
                    thr = float(thrs_grid[j])

            cand = Candidate(feat=feat, threshold=float(thr), op=str(op))
            s, _, _, _ = _eval_one_candidate(
                pl_rows, day_id, cols, chosen, cand, n_days, use_numexpr, max_capital_global
            )
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

        # Final lock to grid (authoritative)
        if lock_to_grid and n_quantiles_lock > 0:
            thrs_grid, qs_grid = _lock_threshold_quantile_grid(x_surv, ql, qh, n_quantiles_lock)
            if thrs_grid.size:
                j = int(np.argmin(np.abs(thrs_grid - thr)))
                thr = float(thrs_grid[j])
                best_q = float(qs_grid[j])

        cand = Candidate(best_feat, float(thr), str(best_op))
        cand_s, cand_ar, cand_av, cand_dd = _eval_one_candidate(
            pl_rows, day_id, cols, chosen, cand, n_days, use_numexpr, max_capital_global
        )

        net_improvement = (cand_s - baseline) - (complexity_penalty * (len(chosen) + 1))
        if net_improvement < improvement_eps:
            break

        chosen.append(cand)
        if cand.feat in remaining:
            remaining.remove(cand.feat)
        best_sharpe, best_ann_ret, best_ann_vol, best_max_dd = cand_s, cand_ar, cand_av, cand_dd

        final_mask = (_build_mask_numexpr(cols, chosen) if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy(cols, chosen))
        daily_pl = _aggregate_daily_sum(pl_rows, day_id, final_mask, n_days)
        best_daily = daily_pl / max_capital_global

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

    # If nothing accepted, report baseline
    if best_daily is None:
        best_daily = daily_ret_all
        best_sharpe, best_ann_ret, best_ann_vol = _annualized_sharpe(best_daily)
        best_max_dd = _max_drawdown(best_daily)
    # else:
    #     best_max_dd = _max_drawdown(best_daily)
    max_dd_final, peak_idx, trough_idx = _max_drawdown_with_indices(best_daily)
    best_max_dd = max_dd_final

    crit_list = _to_crit_list(chosen, ndigits=crit_round_digits, combined=True)
    # crit_list = _to_crit_list(chosen, ndigits=4)
    
    logs = {
    "max_capital_global": max_capital_global,
    "bo_trace": bo_logs,
    "drawdown": {                      # <<< NEW
        "max_dd": best_max_dd,         # negative percentage (e.g., -0.23)
        "peak_idx": peak_idx,          # index into best_daily series
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
        sharpe=best_sharpe,
        ann_return=best_ann_ret,
        ann_vol=best_ann_vol,
        max_dd=best_max_dd,
        daily_series=best_daily,
        chosen=chosen,
        crit_list=crit_list,
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
