from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from notebook_utils._data_explore_functions import create_distance

from .validate import require_columns, require_non_empty_df


def _inventory_distance_columns(final_df: pd.DataFrame, category: str) -> List[str]:
    require_columns(final_df, ["distance", "category", "clean name"], "final_df")
    return final_df[
        (final_df["distance"] == True) & (final_df["category"] == category)
    ]["clean name"].tolist()


def engineer_reviewed_features(base_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    static_rvw_001 = base_df.copy()
    require_non_empty_df(static_rvw_001, "static_rvw_001")
    require_columns(
        static_rvw_001,
        [
            "prev_close",
            "day_open",
            "premkt_m_h",
            "atr_ktg",
            "spy_trade_px",
            "spy_atr",
            "vol_acc",
            "vol_core",
            "d_avol250",
            "rvol",
            "spy_rvol",
            "entry_time",
            "exit_time",
            "week_day",
            "month",
            "day_range",
            "spread_nbbo",
            "entry_price",
        ],
        "static_rvw_001",
    )

    symbol_dist_cols = _inventory_distance_columns(final_df, "symbol")
    symbol_dist_cols += ["qtd", "mtd"]
    spy_dist_cols = _inventory_distance_columns(final_df, "spy")

    for column in symbol_dist_cols:
        create_distance("dist", static_rvw_001, "prev_close", column, "atr_ktg")
        create_distance("dist", static_rvw_001, "day_open", "prev_close", "atr_ktg")
        create_distance("dist", static_rvw_001, "day_open", "premkt_m_h", "atr_ktg")

    clean_symbol_dist_cols = _inventory_distance_columns(final_df, "symbol")
    clean_symbol_dist_cols += ["qtd", "mtd"]
    for column in clean_symbol_dist_cols:
        create_distance("pct", static_rvw_001, "prev_close", column, "prev_close")
        create_distance("pct", static_rvw_001, "day_open", "prev_close", "day_open")
        create_distance("pct", static_rvw_001, "day_open", "premkt_m_h", "day_open")

    clean_spy_dist_cols = _inventory_distance_columns(final_df, "spy")
    if "spy_trade_px" in clean_spy_dist_cols:
        clean_spy_dist_cols.remove("spy_trade_px")

    for column in clean_spy_dist_cols:
        create_distance("dist", static_rvw_001, "spy_trade_px", column, "spy_atr")
    for column in clean_spy_dist_cols:
        create_distance("pct", static_rvw_001, "spy_trade_px", column, "spy_atr")

    static_rvw_001["open_vol"] = static_rvw_001["vol_acc"] - static_rvw_001["vol_core"]
    for item in ["d_avol5", "d_avol50", "premkt_vol", "vol_acc", "open_vol"]:
        static_rvw_001["%s_rat" % item] = static_rvw_001[item] / static_rvw_001["d_avol250"]

    # ADDITION: Adding episol to prevent missing when dividing by zero
    epsilon = 1e-6
    ratio = static_rvw_001["rvol"] / (static_rvw_001["spy_rvol"] + epsilon)
    ratio = ratio.clip(lower=0, upper=50)
    static_rvw_001["symb_spy_rvol_rat"] = ratio
    #
    static_rvw_001["entry_hr_dec"] = static_rvw_001["entry_time"].dt.hour + static_rvw_001["entry_time"].dt.minute / 60
    static_rvw_001["exit_hr_dec"] = static_rvw_001["exit_time"].dt.hour + static_rvw_001["exit_time"].dt.minute / 60
    static_rvw_001["entry_hr_dec_to_close"] = 16.00 - static_rvw_001["entry_hr_dec"]
    static_rvw_001["year_day"] = static_rvw_001["entry_time"].dt.dayofyear
    static_rvw_001["week_day_sin"] = np.sin(2 * np.pi * static_rvw_001["week_day"] / 7)
    static_rvw_001["week_day_cos"] = np.cos(2 * np.pi * static_rvw_001["week_day"] / 7)
    static_rvw_001["month_sin"] = np.sin(2 * np.pi * static_rvw_001["month"] / 12)
    static_rvw_001["month_cos"] = np.cos(2 * np.pi * static_rvw_001["month"] / 12)
    static_rvw_001["year_day_sin"] = np.sin(2 * np.pi * static_rvw_001["year_day"] / 365.25)
    static_rvw_001["year_day_cos"] = np.cos(2 * np.pi * static_rvw_001["year_day"] / 365.25)

    static_rvw_001["dist_day_range"] = static_rvw_001["day_range"] / static_rvw_001["atr_ktg"]
    static_rvw_001["spread_perc"] = static_rvw_001["spread_nbbo"] / static_rvw_001["entry_price"]

    prev_close_vars = [
        "mtd",
        "qtd",
        "ytd",
        "px_1m_ago",
        "px_2m_ago",
        "px_3m_ago",
        "px_4m_ago",
        "px_5m_ago",
        "px_6m_ago",
        "px_7m_ago",
        "px_8m_ago",
        "px_9m_ago",
        "px_10m_ago",
        "px_11m_ago",
        "px_12m_ago",
        "px_prev2",
        "px_prev3",
        "px_prev4",
        "px_prev5",
        "px_prev6",
        "px_prev10",
        "prev_high",
        "prev_low",
        "prev_vwap",
    ]
    for item in prev_close_vars:
        static_rvw_001["ret_%s" % item] = (static_rvw_001[item] / static_rvw_001["prev_close"]) - 1

    ma_vars = ["d_ema8", "d_ema20", "d_ema50", "d_ema100", "d_ema200", "ema", "sma", "kama", "kalmar_f"]
    for item in ma_vars:
        static_rvw_001["ret_%s" % item] = (static_rvw_001[item] / static_rvw_001["prev_close"]) - 1

    for item in ["arimax1", "arimax2", "var1", "var2"]:
        static_rvw_001["dumm_%s" % item] = (static_rvw_001[item] > static_rvw_001["prev_close"]).astype(int)

    return static_rvw_001


def summarize_missing_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    summary = (
        df.isna()
        .sum()
        .to_frame("missing")
        .assign(
            dtype=df.dtypes,
            non_null=lambda x: len(df) - x["missing"],
            percent_missing=lambda x: x["missing"] / len(df) * 100,
            unique=df.nunique(),
        )
    )
    missing_list = summary[summary["missing"] > 0].index.tolist()
    return summary, missing_list


def drop_missing_feature_columns(df: pd.DataFrame, missing_list: List[str]) -> pd.DataFrame:
    cols_to_drop = [col for col in missing_list if col in df.columns]
    cleaned = df.drop(columns=cols_to_drop).copy()
    require_non_empty_df(cleaned, "partition_a")
    return cleaned
