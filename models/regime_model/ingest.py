from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from notebook_utils._dashboard_functions_one_symbol_v2 import dashboard
from notebook_utils._formatting_functions import enrich_data_dictionary, event_import, generate_data_dictionary

from .config import RunConfig
from .io import RegimeModelPaths
from .validate import require_columns, require_file, require_non_empty_df


REQUIRED_GROUP_LISTS = [
    "front_cols",
    "kite_cols",
    "sym_ta_cols",
    "extra_sym_cols",
    "drop_cols",
]


def build_inventory_dataframes(config: RunConfig, paths: RegimeModelPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    first_raw_file = config.raw_event_filenames[0]
    require_file(paths.raw_input_files[first_raw_file], "raw event data file")
    require_file(paths.master_inventory_path, "master dictionary")

    dictionary_df = generate_data_dictionary(
        input_filename=first_raw_file,
        input_folder=paths.raw_input_dir,
        output_folder=paths.intermediate_dir,
    )
    if dictionary_df is None or dictionary_df.empty:
        raise ValueError("Data dictionary generation failed or returned an empty dataframe.")

    final_df = enrich_data_dictionary(
        current_dictionary_df=dictionary_df,
        master_inventory_path=paths.master_inventory_path,
        output_folder=paths.intermediate_dir,
    )
    if final_df is None or final_df.empty:
        raise ValueError("Enriched data dictionary is empty. Review the master inventory mapping.")

    require_columns(final_df, ["code", "kite name"], "final_df")
    return dictionary_df, final_df


def resolve_column_groups(final_df: pd.DataFrame) -> Dict[str, list]:
    grouped = final_df.groupby("code")["kite name"].apply(list).to_dict()
    missing_group_lists = [name for name in REQUIRED_GROUP_LISTS if grouped.get(name) is None]
    if missing_group_lists:
        raise ValueError("Missing expected column groups from master dictionary: %s" % missing_group_lists)

    for name in REQUIRED_GROUP_LISTS + ["last_cols", "extra_cols", "sec_cols"]:
        grouped.setdefault(name, [])

    grouped["front_cols"] = list(grouped["front_cols"])
    grouped["front_cols"].sort(reverse=True)
    return grouped


def load_raw_event_data(config: RunConfig, paths: RegimeModelPaths, grouped_columns: Dict[str, list]) -> pd.DataFrame:
    event_frames = []

    for filename in config.raw_event_filenames:
        require_file(paths.raw_input_files[filename], "raw event data file")
        new_data = event_import(
            paths.raw_input_dir,
            filename,
            grouped_columns["front_cols"],
            grouped_columns["kite_cols"],
            grouped_columns["sym_ta_cols"],
            grouped_columns["extra_sym_cols"],
            grouped_columns["extra_cols"],
            grouped_columns["sec_cols"],
            None,
            grouped_columns["drop_cols"],
        )
        require_columns(new_data, ["normed_date", "entry_time", "exit_time", "symbol", "mtm_pl"], filename)
        event_frames.append(new_data)

    event_data = pd.concat(event_frames, ignore_index=False)
    require_non_empty_df(event_data, "event_data")
    event_data.sort_values(by=["entry_time", "symbol"], ascending=[False, True], inplace=True)
    event_data["modelspec"] = None
    return event_data


def build_dashboard_dataset(config: RunConfig, event_data: pd.DataFrame):
    require_non_empty_df(event_data, "event_data")
    require_columns(
        event_data,
        ["entry_price", "exit_price", "exit_shares", "exit_side", "matched_shares", "mtm_pl"],
        "event_data",
    )

    _, pnl_symboldate_000, pnl_bydate_000 = dashboard(
        event_data,
        config.entry_fee,
        "no_optmz",
        config.long_short,
        "complete",
        "max",
    )
    require_non_empty_df(pnl_symboldate_000, "pnl_symboldate_000")
    require_non_empty_df(pnl_bydate_000, "pnl_bydate_000")
    return pnl_symboldate_000, pnl_bydate_000
