from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError("Missing %s: %s" % (label, path))
    return path


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing %s files: %s" % (label, missing))


def require_non_empty_df(df: pd.DataFrame, df_name: str) -> None:
    if df is None or df.empty:
        raise ValueError("%s is empty. Check the upstream stage or source artifact." % df_name)


def require_columns(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError("%s is missing required columns: %s" % (df_name, missing_columns))


def require_non_empty_list(values: Sequence[object], label: str) -> None:
    if not values:
        raise ValueError("%s is empty. Review the upstream feature-selection stage." % label)


def require_valid_split_index(length: int, split_index: int, label: str) -> None:
    if split_index <= 0 or split_index >= length:
        raise ValueError(
            "Invalid %s split index %s for dataframe length %s." % (label, split_index, length)
        )


def require_matching_lengths(expected_length: int, actual_length: int, label: str) -> None:
    if int(expected_length) != int(actual_length):
        raise ValueError(
            "Length mismatch for %s: expected %s, got %s." % (label, expected_length, actual_length)
        )


def require_export_ready(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    require_non_empty_df(df, df_name)
    require_columns(df, required_columns, df_name)
