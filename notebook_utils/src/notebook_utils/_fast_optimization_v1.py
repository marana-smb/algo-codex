# file: greedy_search_with_returns_and_adaptive.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    feat: str
    threshold: float
    op: str


@dataclass
class SearchResult:
    sharpe: float
    ann_return: float
    ann_vol: float
    max_dd: float
    daily_series: np.ndarray
    chosen: List[Candidate]
    crit_list: List[Dict[str, Tuple[float, str]]]
    trade_count: int
    capital_used: float
    logs: Dict[str, object]


def _to_day_ids(df: pd.DataFrame, date_col: str = "normed_date") -> Tuple[np.ndarray, np.ndarray]:
    dts = pd.to_datetime(df[date_col], errors="coerce")
    try:
        ns = dts.view("int64")
    except Exception:
        ns = dts.astype("int64")
    days = ns // 86_400_000_000_000
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
    return ann_ret / (ann_vol + eps), ann_ret, ann_vol


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
    x = daily_returns.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, -1, -1
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq[: trough_idx + 1]))
    return float(dd.min()), peak_idx, trough_idx


def _aggregate_daily_sum(values: np.ndarray, day_id: np.ndarray, mask: Optional[np.ndarray], n_days: int) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(day_id, dtype=bool)
    good = mask & np.isfinite(values)
    return np.bincount(day_id[good], weights=values[good], minlength=n_days).astype(np.float64, copy=False)


def _estimate_capital_proxy_rows(
    df: pd.DataFrame,
    *,
    mtm_col: str = "mtm_pl",
    entry_shares_col: str = "entry_shares",
    entry_price_col: str = "entry_price",
    matched_shares_col: str = "matched_shares",
    exit_shares_col: str = "exit_shares",
    exit_price_col: str = "exit_price",
) -> np.ndarray:
    if entry_shares_col in df and entry_price_col in df:
        cap_row = np.abs(
            df[entry_shares_col].to_numpy(dtype=np.float64, copy=False)
            * df[entry_price_col].to_numpy(dtype=np.float64, copy=False)
        )
    elif matched_shares_col in df and entry_price_col in df:
        cap_row = np.abs(
            df[matched_shares_col].to_numpy(dtype=np.float64, copy=False)
            * df[entry_price_col].to_numpy(dtype=np.float64, copy=False)
        )
    elif exit_shares_col in df and exit_price_col in df:
        cap_row = np.abs(
            df[exit_shares_col].to_numpy(dtype=np.float64, copy=False)
            * df[exit_price_col].to_numpy(dtype=np.float64, copy=False)
        )
    else:
        warnings.warn("Could not infer capital proxy; falling back to |mtm_pl| for scale (rough).")
        cap_row = np.abs(df[mtm_col].to_numpy(dtype=np.float64, copy=False))
    cap_row = np.asarray(cap_row, dtype=np.float64)
    cap_row[~np.isfinite(cap_row)] = 0.0
    return cap_row


def _estimate_max_capital_global(
    df: pd.DataFrame,
    day_id: np.ndarray,
    n_days: int,
    **kwargs: str,
) -> float:
    cap_row = _estimate_capital_proxy_rows(df, **kwargs)
    cap_daily = _aggregate_daily_sum(cap_row, day_id, mask=None, n_days=n_days)
    max_cap = float(np.nanmax(cap_daily)) if cap_daily.size else 0.0
    if not np.isfinite(max_cap) or max_cap <= 0.0:
        warnings.warn("Estimated max capital <= 0; Sharpe on returns may be ill-defined. Using 1.0.")
        max_cap = 1.0
    return max_cap


def _masked_capital_used(cap_row: np.ndarray, day_id: np.ndarray, mask: np.ndarray, n_days: int) -> float:
    cap_daily = _aggregate_daily_sum(cap_row, day_id, mask, n_days)
    capital_used = float(np.nanmax(cap_daily)) if cap_daily.size else 0.0
    if not np.isfinite(capital_used) or capital_used <= 0.0:
        capital_used = 1.0
    return capital_used


def _build_mask_numpy(cols: Dict[str, np.ndarray], chosen: List[Candidate]) -> np.ndarray:
    if not chosen:
        n = next(iter(cols.values())).shape[0]
        return np.ones(n, dtype=bool)
    n = next(iter(cols.values())).shape[0]
    mask = np.ones(n, dtype=bool)
    for candidate in chosen:
        x = cols[candidate.feat]
        thr = np.asarray(candidate.threshold, dtype=x.dtype)
        pred = (x >= thr) if candidate.op == ">=" else (x <= thr)
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
    if not chosen:
        n = next(iter(cols.values())).shape[0]
        return np.ones(n, dtype=bool)

    expr_parts: List[str] = []
    local_dict: Dict[str, Union[np.ndarray, float, bool]] = {}
    for i, candidate in enumerate(chosen):
        xi, ti, fi = f"x{i}", f"t{i}", f"f{i}"
        x = cols[candidate.feat]
        local_dict[xi] = x
        local_dict[ti] = float(candidate.threshold)
        local_dict[fi] = np.isfinite(x)
        expr_parts.append(f"({fi}) & ({xi}{candidate.op}{ti})")

    mask = ne.evaluate("&".join(expr_parts), local_dict=local_dict)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    return mask


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
        return 0.25, 0.75
    frac = min(0.5, max(0.0, min_count_per_side / float(n_survive)))
    low = max(base_low, frac)
    high = min(base_high, 1.0 - frac)
    if low >= high:
        return 0.25, 0.75
    return low, high


def _make_trials(
    feature_names: List[str],
    cols: Dict[str, np.ndarray],
    n_quantiles: int,
    directions: Tuple[str, str] = (">=", "<="),
    rng: Optional[np.random.Generator] = None,
    max_trials_per_feat: Optional[int] = 48,
) -> Tuple[Dict[str, List[Candidate]], List[Dict[str, Union[int, str]]]]:
    out: Dict[str, List[Candidate]] = {}
    logs: List[Dict[str, Union[int, str]]] = []
    for feature_name in feature_names:
        thrs = _quantile_thresholds(cols[feature_name], n_quantiles)
        total = int(len(thrs) * len(directions))
        cands = [Candidate(feature_name, float(threshold), op) for op in directions for threshold in thrs]
        sampled = total
        if max_trials_per_feat and rng is not None and len(cands) > max_trials_per_feat:
            idx = rng.choice(len(cands), size=max_trials_per_feat, replace=False)
            cands = [cands[i] for i in idx]
            sampled = int(len(cands))
        out[feature_name] = cands
        logs.append({"feature": feature_name, "total": total, "sampled": sampled})
    return out, logs


def _evaluate_mask_metrics(
    daily_pl: np.ndarray,
    cap_row: np.ndarray,
    day_id: np.ndarray,
    mask: np.ndarray,
    n_days: int,
) -> Dict[str, Union[float, int, np.ndarray]]:
    trade_count = int(mask.sum())
    if trade_count <= 0:
        empty_daily = np.zeros(n_days, dtype=np.float64)
        return {
            "trade_count": 0,
            "capital_used": 1.0,
            "daily_pl": empty_daily,
            "daily_ret": empty_daily,
            "sharpe": -np.inf,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "max_dd": 0.0,
        }

    daily_sum = _aggregate_daily_sum(daily_pl, day_id, mask, n_days)
    capital_used = _masked_capital_used(cap_row, day_id, mask, n_days)
    daily_ret = daily_sum / capital_used
    sharpe, ann_return, ann_vol = _annualized_sharpe(daily_ret)
    max_dd = _max_drawdown(daily_ret)
    return {
        "trade_count": trade_count,
        "capital_used": capital_used,
        "daily_pl": daily_sum,
        "daily_ret": daily_ret,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
    }


def _eval_candidates_fast(
    daily_pl: np.ndarray,
    cap_row: np.ndarray,
    day_id: np.ndarray,
    cols: Dict[str, np.ndarray],
    base_chosen: List[Candidate],
    trial_set: Iterable[Candidate],
    n_days: int,
    use_numexpr: bool,
) -> List[Tuple[Candidate, Dict[str, Union[float, int, np.ndarray]]]]:
    out: List[Tuple[Candidate, Dict[str, Union[float, int, np.ndarray]]]] = []
    build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
    for cand in trial_set:
        mask = build(cols, base_chosen + [cand])
        out.append((cand, _evaluate_mask_metrics(daily_pl, cap_row, day_id, mask, n_days)))
    return out


def _to_crit_list(chosen: List[Candidate], ndigits: int = 4, combined: bool = False) -> List[Dict[str, Tuple[float, str]]]:
    if combined:
        return [{candidate.feat: (round(float(candidate.threshold), ndigits), candidate.op) for candidate in chosen}]
    return [{candidate.feat: (round(float(candidate.threshold), ndigits), candidate.op)} for candidate in chosen]


def crit_list_to_candidates(
    crit_list: Union[Dict[str, Tuple[float, str]], List[Dict[str, Tuple[float, str]]], None]
) -> List[Candidate]:
    if crit_list is None:
        return []
    items = [crit_list] if isinstance(crit_list, dict) else crit_list
    chosen: List[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for feat, value in item.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                threshold, op = value
                chosen.append(Candidate(str(feat), float(threshold), str(op)))
    return chosen


def evaluate_threshold_rule(
    df: pd.DataFrame,
    crit_list: Union[Dict[str, Tuple[float, str]], List[Dict[str, Tuple[float, str]]], None],
    *,
    mtm_col: str = "mtm_pl",
    date_col: str = "normed_date",
    use_numexpr: bool = True,
    crit_round_digits: int = 4,
) -> Dict[str, Union[float, int, np.ndarray, Dict[str, Tuple[float, str]], List[Dict[str, Tuple[float, str]]]]]:
    chosen = crit_list_to_candidates(crit_list)
    day_id, uniq_days = _to_day_ids(df, date_col=date_col)
    n_days = int(uniq_days.size)
    pl_rows = df[mtm_col].astype(np.float64).to_numpy(copy=False)
    cap_row = _estimate_capital_proxy_rows(df, mtm_col=mtm_col)
    cols = {candidate.feat: df[candidate.feat].astype(np.float32).to_numpy(copy=False) for candidate in chosen}
    build = _build_mask_numexpr if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy
    mask = build(cols, chosen) if chosen else np.ones(len(df), dtype=bool)
    metrics = _evaluate_mask_metrics(pl_rows, cap_row, day_id, mask, n_days)
    metrics["mask"] = mask
    metrics["rule_dict"] = {candidate.feat: (candidate.threshold, candidate.op) for candidate in chosen}
    metrics["crit_list"] = _to_crit_list(chosen, ndigits=crit_round_digits, combined=True)
    metrics["max_capital_global"] = _estimate_max_capital_global(df, day_id, n_days, mtm_col=mtm_col)
    return metrics


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
    adaptive_tails: bool = True,
    base_q_low: float = 0.05,
    base_q_high: float = 0.95,
    min_count_per_side: int = 200,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    pl_rows = df[mtm_col].astype(np.float64).to_numpy(copy=False)
    day_id, uniq_days = _to_day_ids(df, date_col=date_col)
    n_days = int(uniq_days.size)
    cap_row = _estimate_capital_proxy_rows(df, mtm_col=mtm_col)
    cols: Dict[str, np.ndarray] = {feature_name: df[feature_name].astype(np.float32).to_numpy(copy=False) for feature_name in features}

    max_capital_global = _estimate_max_capital_global(df, day_id, n_days, mtm_col=mtm_col)
    baseline_metrics = _evaluate_mask_metrics(pl_rows, cap_row, day_id, np.ones(len(df), dtype=bool), n_days)
    base_sharpe = float(baseline_metrics["sharpe"])
    trials_by_feat, trials_log = _make_trials(features, cols, n_quantiles, rng=rng, max_trials_per_feat=random_subset_per_feat)

    chosen: List[Candidate] = []
    best_metrics = baseline_metrics
    best_daily: Optional[np.ndarray] = None
    remaining = set(features)

    for _step in range(max_features):
        base_mask = (_build_mask_numexpr(cols, chosen) if (use_numexpr and HAS_NUMEXPR) else _build_mask_numpy(cols, chosen))
        best_local: Optional[Tuple[Candidate, Dict[str, Union[float, int, np.ndarray]]]] = None

        for feature_name in list(remaining):
            if adaptive_tails:
                x_sub = cols[feature_name][base_mask]
                n_sub = int(np.isfinite(x_sub).sum())
                low, high = _adaptive_bounds(n_sub, base_q_low, base_q_high, min_count_per_side)
                thrs = _quantile_thresholds(x_sub, n_quantiles, q_low=low, q_high=high)
                if thrs.size == 0:
                    evals = []
                else:
                    cands = [Candidate(feature_name, float(threshold), op) for op in (">=", "<=") for threshold in thrs]
                    if random_subset_per_feat and len(cands) > random_subset_per_feat:
                        idx = rng.choice(len(cands), size=random_subset_per_feat, replace=False)
                        cands = [cands[i] for i in idx]
                    evals = _eval_candidates_fast(pl_rows, cap_row, day_id, cols, chosen, cands, n_days, use_numexpr)
            else:
                evals = _eval_candidates_fast(pl_rows, cap_row, day_id, cols, chosen, trials_by_feat[feature_name], n_days, use_numexpr)

            cand_local = max(evals, key=lambda item: float(item[1]["sharpe"])) if evals else None
            if cand_local and (best_local is None or float(cand_local[1]["sharpe"]) > float(best_local[1]["sharpe"])):
                best_local = cand_local

        if best_local is None:
            break

        candidate, candidate_metrics = best_local
        candidate_sharpe = float(candidate_metrics["sharpe"])
        baseline = float(best_metrics["sharpe"]) if chosen else base_sharpe
        if candidate_sharpe < baseline + improvement_eps:
            break

        chosen.append(candidate)
        remaining.discard(candidate.feat)
        best_metrics = candidate_metrics
        best_daily = np.asarray(candidate_metrics["daily_ret"], dtype=np.float64)

    if best_daily is None:
        best_daily = np.asarray(baseline_metrics["daily_ret"], dtype=np.float64)
        best_metrics = baseline_metrics

    crit_list = _to_crit_list(chosen, ndigits=crit_round_digits, combined=True)
    logs = {
        "max_capital_global": max_capital_global,
        "capital_used": float(best_metrics["capital_used"]),
        "trade_count": int(best_metrics["trade_count"]),
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
