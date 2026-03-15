"""Target construction for the regime model."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_COLUMNS = {
    "mtm_pl",
    "pl_g",
    "pl_n",
    "fees",
    "Capital",
    "target_down",
    "target_return",
    "wins",
}


def create_regime_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the downside target without leaking future outcome columns into features."""
    out = df.copy()
    direction = np.where(out["entry_side"].fillna(0) < 0, 1, -1)
    out["target_down"] = (direction * out["pl_n"].fillna(0.0) > 0).astype(int)
    out["target_return"] = out["pl_n"] / out["Capital"]
    return out


def drop_unusable_rows(df: pd.DataFrame, feature_columns: Iterable[str], target_column: str = "target_down") -> pd.DataFrame:
    """Remove rows that cannot be used for model fitting."""
    required_columns = list(feature_columns) + [target_column, "normed_date", "entry_time"]
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_columns).copy()
    clean_df.sort_values(["normed_date", "entry_time", "symbol"], inplace=True)
    clean_df.reset_index(drop=True, inplace=True)
    return clean_df


def validate_feature_columns(feature_columns: Iterable[str]) -> List[str]:
    """Validate that the selected features do not contain target leakage."""
    safe_columns = []
    for column in feature_columns:
        if column in FORBIDDEN_FEATURE_COLUMNS:
            raise ValueError("Leaky feature detected: %s" % column)
        safe_columns.append(column)
    return safe_columns
