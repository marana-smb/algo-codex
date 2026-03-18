from __future__ import annotations

import pandas as pd

from .io import RegimeModelPaths
from .validate import require_columns, require_file, require_non_empty_df


LIQUIDITY_USECOLS = [
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


def merge_liquidity_features(df: pd.DataFrame, paths: RegimeModelPaths) -> pd.DataFrame:
    require_file(paths.liquidity_path, "liquidity research file")
    require_columns(df, ["normed_date"], "df_before_liquidity_merge")

    liquidity_df = pd.read_excel(
        paths.liquidity_path,
        sheet_name="Sheet1",
        usecols=LIQUIDITY_USECOLS,
        parse_dates=["DATE"],
    )
    liquidity_df = liquidity_df.dropna(how="any")
    require_non_empty_df(liquidity_df, "liquidity_df")
    liquidity_df["normed_date"] = liquidity_df["DATE"].dt.strftime("%Y-%m-%d")

    merged = df.copy()
    merged["normed_date"] = pd.to_datetime(merged["normed_date"]).dt.normalize()
    liquidity_df["normed_date"] = pd.to_datetime(liquidity_df["normed_date"]).dt.normalize()

    merged = merged.merge(
        liquidity_df.drop(columns=["DATE"]),
        how="left",
        on="normed_date",
        suffixes=("", "_liq"),
    )
    require_columns(merged, ["PCA_Index_full"], "static_rvw_002")
    return merged


def merge_fear_greed(df: pd.DataFrame, paths: RegimeModelPaths) -> pd.DataFrame:
    require_file(paths.fear_greed_path, "fear and greed research file")
    require_columns(df, ["normed_date"], "df_before_fear_greed_merge")

    fear_df = pd.read_csv(paths.fear_greed_path, usecols=["date", "value"])
    fear_df = fear_df.dropna(how="any")
    require_non_empty_df(fear_df, "fear_df")
    fear_df = fear_df.rename(columns={"date": "normed_date", "value": "fear_greed"})
    fear_df["normed_date"] = pd.to_datetime(fear_df["normed_date"]).dt.normalize()

    merged = df.copy()
    merged["normed_date"] = pd.to_datetime(merged["normed_date"]).dt.normalize()
    merged = merged.merge(
        fear_df,
        how="left",
        on="normed_date",
        suffixes=("", "_liq"),
    )
    require_columns(merged, ["fear_greed"], "static_rvw_003")
    return merged
