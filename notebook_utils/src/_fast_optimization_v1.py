# file: greedy_search_with_returns_and_adaptive.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

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

def _max_drawdown(daily_returns: np.ndarray) -> float:
    eq = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())

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

# ---------------- Trials ----------------
def _make_trials(
    feature_names: List[str],
    cols: Dict[str, np.ndarray],
    n_quantiles: int,
    directions: Tuple[str, str] = (">=", "<="),
    rng: Optional[np.random.Generator] = None,
    max_trials_per_feat: Optional[int] = 48,
) -> Tuple[Dict[str, List[Candidate]], List[Dict[str, Union[int, str]]]]:
    """
    Build static trials and log counts.
    Returns:
      (trials_by_feat, trials_log)
      trials_log entries: {'feature': str, 'total': int, 'sampled': int}
    """
    out: Dict[str, List[Candidate]] = {}
    logs: List[Dict[str, Union[int, str]]] = []

    for f in feature_names:
        thrs = _quantile_thresholds(cols[f], n_quantiles)
        total = int(len(thrs) * len(directions))
        cands = [Candidate(f, float(t), op) for op in directions for t in thrs]

        sampled = total
        if max_trials_per_feat and rng is not None and len(cands) > max_trials_per_feat:
            idx = rng.choice(len(cands), size=max_trials_per_feat, replace=False)
            cands = [cands[i] for i in idx]
            sampled = int(len(cands))

        out[f] = cands
        logs.append({"feature": f, "total": total, "sampled": sampled})

    return out, logs

# ---------------- Evaluation ----------------
def _eval_candidates_fast(
    daily_pl: np.ndarray,
    day_id: np.ndarray,
    cols: Dict[str, np.ndarray],
    base_chosen: List[Candidate],
    trial_set: Iterable[Candidate],
    n_days: int,
    use_numexpr: bool,
    *,
    denominator: float,  # global max capital
) -> List[Tuple[Candidate, float, float, float, float]]:
    out: List[Tuple[Candidate, float, float, float, float]] = []
    build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
    for cand in trial_set:
        chosen = base_chosen + [cand]
        mask = build(cols, chosen)
        if not mask.any():
            out.append((cand, -np.inf, 0.0, 0.0, 0.0))
            continue
        daily_sum = _aggregate_daily_sum(daily_pl, day_id, mask, n_days)
        daily_ret = daily_sum / denominator
        s, ar, av = _annualized_sharpe(daily_ret)
        dd = _max_drawdown(daily_ret)
        out.append((cand, s, ar, av, dd))
    return out

# ---------------- Orchestrator ----------------
def _to_crit_list(chosen: List[Candidate], ndigits: int = 4) -> List[Dict[str, Tuple[float, str]]]:
    return [{c.feat: (round(float(c.threshold), ndigits), c.op)} for c in chosen]

def greedy_threshold_search(
    df: pd.DataFrame,
    features: List[str],
    *,
    mtm_col: str = "mtm_pl",
    date_col: str = "normed_date",
    n_quantiles: int = 24,
    max_features: int = 6,
    improvement_eps: float = 0.02,
    seed: int = 42,
    use_numexpr: bool = True,
    random_subset_per_feat: Optional[int] = 48,
    crit_round_digits: int = 4,
    # NEW:
    adaptive_tails: bool = True,
    base_q_low: float = 0.05,
    base_q_high: float = 0.95,
    min_count_per_side: int = 200,
) -> SearchResult:
    rng = np.random.default_rng(seed)

    # Materialize arrays
    pl_rows = df[mtm_col].astype(np.float64).to_numpy(copy=False)
    day_id, uniq_days = _to_day_ids(df, date_col=date_col)
    n_days = int(uniq_days.size)
    cols: Dict[str, np.ndarray] = {f: df[f].astype(np.float32).to_numpy(copy=False) for f in features}

    # Global denominator: max capital across days
    max_capital_global = _estimate_max_capital_global(df, day_id, n_days)

    # Static trials + logging
    trials_by_feat, trials_log = _make_trials(
        features, cols, n_quantiles, rng=rng, max_trials_per_feat=random_subset_per_feat
    )

    # Baseline (no filters) on RETURNS
    daily_pl_all = _aggregate_daily_sum(pl_rows, day_id, mask=None, n_days=n_days)
    daily_ret_all = daily_pl_all / max_capital_global
    base_sharpe, _, _ = _annualized_sharpe(daily_ret_all)

    chosen: List[Candidate] = []
    best_sharpe = -np.inf
    best_ann_ret = best_ann_vol = best_max_dd = 0.0
    best_daily: Optional[np.ndarray] = None
    remaining = set(features)

    for _step in range(max_features):
        # Build mask from already-accepted predicates once (for adaptive tails)
        base_mask = (_build_mask_numexpr(cols, chosen) if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy(cols, chosen))

        best_local = None  # (Candidate, s, ar, av, dd)

        for f in list(remaining):
            # Optionally recompute thresholds adaptively on surviving sample for feature f
            if adaptive_tails:
                x_sub = cols[f][base_mask]
                n_sub = int(np.isfinite(x_sub).sum())
                low, high = _adaptive_bounds(n_sub, base_q_low, base_q_high, min_count_per_side)
                thrs = _quantile_thresholds(x_sub, n_quantiles, q_low=low, q_high=high)
                if thrs.size == 0:
                    evals = []
                else:
                    cands = [Candidate(f, float(t), op) for op in (">=", "<=") for t in thrs]
                    # Optional re-sampling if a cap exists
                    if random_subset_per_feat and len(cands) > random_subset_per_feat:
                        idx = rng.choice(len(cands), size=random_subset_per_feat, replace=False)
                        cands = [cands[i] for i in idx]
                    evals = _eval_candidates_fast(
                        pl_rows, day_id, cols, chosen, cands, n_days, use_numexpr, denominator=max_capital_global
                    )
            else:
                evals = _eval_candidates_fast(
                    pl_rows, day_id, cols, chosen, trials_by_feat[f], n_days, use_numexpr, denominator=max_capital_global
                )

            cand_local = max(evals, key=lambda z: z[1]) if evals else None
            if cand_local and (best_local is None or cand_local[1] > best_local[1]):
                best_local = cand_local

        if best_local is None:
            break

        cand, cand_s, cand_ar, cand_av, cand_dd = best_local
        baseline = best_sharpe if chosen else base_sharpe
        if cand_s < baseline + improvement_eps:
            break  # early stop

        # Accept
        chosen.append(cand)
        remaining.discard(cand.feat)
        best_sharpe, best_ann_ret, best_ann_vol, best_max_dd = cand_s, cand_ar, cand_av, cand_dd

        # Store best daily returns curve for reporting
        mask = (_build_mask_numexpr(cols, chosen) if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy(cols, chosen))
        daily_pl = _aggregate_daily_sum(pl_rows, day_id, mask, n_days)
        best_daily = (daily_pl / max_capital_global)

    # Fallback if nothing accepted
    if best_daily is None:
        best_daily = daily_ret_all
        best_sharpe, best_ann_ret, best_ann_vol = _annualized_sharpe(best_daily)
        best_max_dd = _max_drawdown(best_daily)
    else:
        best_max_dd = _max_drawdown(best_daily)

    crit_list = _to_crit_list(chosen, ndigits=crit_round_digits)
    logs = {
        "max_capital_global": max_capital_global,
        "trials_per_feature": trials_log,
        "settings": {
            "adaptive_tails": adaptive_tails,
            "base_q_low": base_q_low,
            "base_q_high": base_q_high,
            "min_count_per_side": min_count_per_side,
            "n_quantiles": n_quantiles,
            "random_subset_per_feat": random_subset_per_feat,
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
