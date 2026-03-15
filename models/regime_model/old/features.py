"""Feature engineering for the regime model pipeline."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from notebook_utils._formatting_functions import nan_inf_summary

EXTERNAL_FEATURE_COLUMNS = [
    "fear_greed",
    "dumm_fear",
    "dumm_greed",
    "PCA_Index_full",
    "entry_hr_dec",
    "entry_hr_dec_to_close",
    "week_day_sin",
    "week_day_cos",
    "month_sin",
    "month_cos",
    "year_day_sin",
    "year_day_cos",
]

LEAKY_COLUMNS = {
    "mtm_pl",
    "pl_g",
    "pl_n",
    "fees",
    "Capital",
    "target_down",
    "target_return",
    "wins",
}


@lru_cache(maxsize=4)
def load_master_variable_inventory(inventory_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the authoritative raw-schema and optimization inventory sheets."""
    path = Path(inventory_path)
    master_df = pd.read_excel(path, sheet_name="master")
    optimization_df = pd.read_excel(path, sheet_name="optimization")
    return master_df, optimization_df


def _build_rename_map(master_df: pd.DataFrame) -> Dict[str, str]:
    rename_df = master_df[["kite name", "clean name"]].dropna().copy()
    rename_df["kite name"] = rename_df["kite name"].astype(str)
    rename_df["clean name"] = rename_df["clean name"].astype(str)
    rename_df = rename_df.drop_duplicates(subset=["kite name"], keep="last")
    return dict(zip(rename_df["kite name"], rename_df["clean name"]))


def load_raw_backtest_data(input_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate raw backtest files for a model round."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError("No raw backtest CSV files found at %s" % input_dir)

    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    raw_df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return raw_df


def normalize_raw_backtest(raw_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw columns using the authoritative master variable inventory."""
    df = raw_df.copy()
    df.rename(columns=_build_rename_map(master_df), inplace=True)

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["normed_date"] = df["entry_time"].dt.normalize()
    df["last_px"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["fees"] = pd.to_numeric(df["entry_fees"], errors="coerce").fillna(0.0) + pd.to_numeric(
        df["exit_fees"], errors="coerce"
    ).fillna(0.0)
    df["pl_g"] = pd.to_numeric(df["mtm_pl"], errors="coerce")
    df["pl_n"] = df["pl_g"] - df["fees"]
    df["Capital"] = (pd.to_numeric(df["entry_price"], errors="coerce").abs() * pd.to_numeric(
        df["matched_shares"], errors="coerce"
    )).replace(0, np.nan)
    df["wins"] = (df["pl_n"] > 0).astype(int)
    df["modelspec"] = None

    sentinel_values = [-9999.0, -9999, -999.99, 9999.0, 9999]
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].replace(sentinel_values, np.nan)

    df.sort_values(["entry_time", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _safe_distance(numerator: pd.Series, denominator: pd.Series, scale: pd.Series) -> pd.Series:
    scale = scale.replace(0, np.nan)
    return (numerator - denominator) / scale


def _merge_liquidity_features(df: pd.DataFrame, liquidity_path: Path) -> pd.DataFrame:
    cols = [
        "DATE",
        "PCA_Index_ma5",
        "PCA_ScaledIndex_ma5",
        "PCA_Index_ma20",
        "PCA_ScaledIndex_ma20",
        "PCA_Index_ma50",
        "PCA_ScaledIndex_ma50",
        "PCA_Raw_full",
        "PCA_Index_full",
    ]
    liquidity_df = pd.read_excel(liquidity_path, sheet_name="Sheet1", usecols=cols, parse_dates=["DATE"])
    liquidity_df = liquidity_df.dropna(how="any").copy()
    liquidity_df["normed_date"] = liquidity_df["DATE"].dt.normalize()
    return df.merge(liquidity_df.drop(columns=["DATE"]), on="normed_date", how="left")


def _merge_fear_greed(df: pd.DataFrame, fear_greed_path: Path) -> pd.DataFrame:
    fear_df = pd.read_csv(fear_greed_path, usecols=["date", "value"])
    fear_df = fear_df.dropna(how="any").rename(columns={"date": "normed_date", "value": "fear_greed"})
    fear_df["normed_date"] = pd.to_datetime(fear_df["normed_date"], errors="coerce").dt.normalize()
    return df.merge(fear_df, on="normed_date", how="left")


def engineer_regime_features(
    df: pd.DataFrame,
    *,
    liquidity_path: Optional[Path] = None,
    fear_greed_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Create the reusable feature set used by the regime model."""
    featured = df.copy()

    if liquidity_path is not None and liquidity_path.exists():
        featured = _merge_liquidity_features(featured, liquidity_path)
    if fear_greed_path is not None and fear_greed_path.exists():
        featured = _merge_fear_greed(featured, fear_greed_path)

    featured["dumm_fear"] = (featured["fear_greed"] < 30).astype(int)
    featured["dumm_greed"] = (featured["fear_greed"] > 70).astype(int)

    if {"last_px", "prev_close", "atr_250"}.issubset(featured.columns):
        featured["dist_last_px_prev_close"] = _safe_distance(featured["last_px"], featured["prev_close"], featured["atr_250"])
    if {"last_px", "prev_high", "atr_250"}.issubset(featured.columns):
        featured["dist_last_px_prev_high"] = _safe_distance(featured["last_px"], featured["prev_high"], featured["atr_250"])
    if {"last_px", "prev_low", "atr_250"}.issubset(featured.columns):
        featured["dist_last_px_prev_low"] = _safe_distance(featured["last_px"], featured["prev_low"], featured["atr_250"])
    if {"last_px", "prev_vwap", "atr_250"}.issubset(featured.columns):
        featured["dist_last_px_prev_vwap"] = _safe_distance(featured["last_px"], featured["prev_vwap"], featured["atr_250"])

    for lag_col in ["px_prev2", "px_prev3", "px_prev4", "px_prev5", "px_prev6", "px_prev10"]:
        if {lag_col, "prev_close"}.issubset(featured.columns):
            featured["ret_%s" % lag_col] = (featured[lag_col] / featured["prev_close"]) - 1.0

    prev_close_ret_vars = sorted([col for col in featured.columns if col.startswith("ret_px_prev")])
    if prev_close_ret_vars:
        featured["pos_ret"] = (featured[prev_close_ret_vars] > 0).sum(axis=1)
        weights = np.array([1.0 / float(col.split("prev")[-1]) for col in prev_close_ret_vars], dtype=float)
        weights = weights / weights.sum()
        weighted = featured[prev_close_ret_vars].gt(0).astype(float).values * weights
        featured["pos_pct"] = weighted.sum(axis=1)

    for pred_col in ["arimax1", "arimax2", "var1", "var2"]:
        if {pred_col, "prev_close"}.issubset(featured.columns):
            featured["dumm_%s" % pred_col] = (featured[pred_col] > featured["prev_close"]).astype(int)

    featured["entry_hr_dec"] = featured["entry_time"].dt.hour + (featured["entry_time"].dt.minute / 60.0)
    featured["exit_hr_dec"] = featured["exit_time"].dt.hour + (featured["exit_time"].dt.minute / 60.0)
    featured["entry_hr_dec_to_close"] = 16.0 - featured["entry_hr_dec"]
    featured["week_day"] = featured["entry_time"].dt.dayofweek
    featured["month"] = featured["entry_time"].dt.month
    featured["year_day"] = featured["entry_time"].dt.dayofyear
    featured["week_day_sin"] = np.sin(2.0 * np.pi * featured["week_day"] / 7.0)
    featured["week_day_cos"] = np.cos(2.0 * np.pi * featured["week_day"] / 7.0)
    featured["month_sin"] = np.sin(2.0 * np.pi * featured["month"] / 12.0)
    featured["month_cos"] = np.cos(2.0 * np.pi * featured["month"] / 12.0)
    featured["year_day_sin"] = np.sin(2.0 * np.pi * featured["year_day"] / 365.25)
    featured["year_day_cos"] = np.cos(2.0 * np.pi * featured["year_day"] / 365.25)

    featured.sort_values(["normed_date", "entry_time", "symbol"], inplace=True)
    featured.reset_index(drop=True, inplace=True)
    return featured


def select_feature_columns(
    df: pd.DataFrame,
    optimization_df: pd.DataFrame,
    candidate_columns: Optional[Iterable[str]] = None,
) -> List[str]:
    """Select feature columns using the authoritative optimization inventory."""
    if candidate_columns is None:
        candidate_columns = optimization_df.loc[optimization_df["optimize"] == True, "clean name"].tolist()
        candidate_columns = list(candidate_columns) + EXTERNAL_FEATURE_COLUMNS

    feature_columns = []
    for column in candidate_columns:
        if column in df.columns and column not in LEAKY_COLUMNS:
            if column not in feature_columns:
                feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No usable feature columns were found for the regime model.")

    return feature_columns


def prepare_feature_frame(
    input_dir: Path,
    *,
    inventory_path: Path,
    liquidity_path: Optional[Path] = None,
    fear_greed_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, normalize, and feature-engineer the raw regime model data."""
    master_df, optimization_df = load_master_variable_inventory(str(inventory_path))
    raw_df = load_raw_backtest_data(input_dir)
    normalized_df = normalize_raw_backtest(raw_df, master_df)
    featured_df = engineer_regime_features(
        normalized_df,
        liquidity_path=liquidity_path,
        fear_greed_path=fear_greed_path,
    )
    return featured_df, master_df, optimization_df


def summarize_data_quality(df: pd.DataFrame, df_name: str = "regime_model") -> pd.DataFrame:
    """Wrap the shared notebook utility for consistent NaN/Inf reporting."""
    return nan_inf_summary(df, print_columns=False, df_name=df_name)
