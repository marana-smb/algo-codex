from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .validate import require_non_empty_df, require_valid_split_index


@dataclass
class SplitBundle:
    ins_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    oos_df: pd.DataFrame
    target_column: str = "wins"
    feature_columns: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "partition_ins": self.ins_df,
            "partition_ins_80": self.train_df,
            "partition_ins_20": self.test_df,
            "partition_oos": self.oos_df,
        }


def filter_by_start_date(df: pd.DataFrame, start_date: Optional[str], date_col: str = "normed_date") -> pd.DataFrame:
    if not start_date:
        return df.copy()
    filtered = df[pd.to_datetime(df[date_col]) >= pd.Timestamp(start_date)].copy()
    return filtered


def split_by_breakdate_fraction(df: pd.DataFrame, fraction: float, date_col: str = "normed_date", label: str = "split"):
    require_non_empty_df(df, label)
    sample = int(round(len(df) * fraction, 0))
    require_valid_split_index(len(df), sample, label)
    break_date = df.iloc[sample][date_col]

    left = df[df[date_col] < break_date].copy()
    right = df[df[date_col] >= break_date].copy()

    require_non_empty_df(left, "%s_left" % label)
    require_non_empty_df(right, "%s_right" % label)
    return left, right, break_date


def create_split_bundle(
    df: pd.DataFrame,
    *,
    insample_fraction: float = 0.80,
    train_fraction: float = 0.80,
    target_column: str = "wins",
    feature_columns: Optional[Iterable[str]] = None,
    date_col: str = "normed_date",
    ins_start_date: Optional[str] = None,
) -> SplitBundle:
    base_df = filter_by_start_date(df, ins_start_date, date_col=date_col)
    require_non_empty_df(base_df, "base_df")

    ins_df, oos_df, _ = split_by_breakdate_fraction(
        base_df,
        insample_fraction,
        date_col=date_col,
        label="ins_oos",
    )
    train_df, test_df, _ = split_by_breakdate_fraction(
        ins_df,
        train_fraction,
        date_col=date_col,
        label="train_test",
    )

    return SplitBundle(
        ins_df=ins_df,
        train_df=train_df,
        test_df=test_df,
        oos_df=oos_df,
        target_column=target_column,
        feature_columns=list(feature_columns) if feature_columns is not None else None,
    )
