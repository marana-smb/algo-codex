from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell
from nbformat.validator import normalize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def set_cell(nb, index, source):
    nb.cells[index]["source"] = source.strip("\n") + "\n"


def clear_code_outputs(nb):
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def write_notebook(source_name, target_name, updates, markdown_inserts=None):
    source_path = NOTEBOOKS_DIR / source_name
    target_path = NOTEBOOKS_DIR / target_name

    nb = nbformat.read(source_path, as_version=4)
    normalize(nb)

    for index, source in updates.items():
        set_cell(nb, index, source)

    if markdown_inserts:
        offset = 0
        for index, markdown_source in markdown_inserts:
            nb.cells.insert(index + offset, new_markdown_cell(markdown_source.strip("\n")))
            offset += 1

    clear_code_outputs(nb)
    nbformat.write(nb, target_path)
    print(f"Wrote {target_path}")


nb1_updates = {
    0: """
# Standard libraries and notebook bootstrap

import ast
import gc
import importlib
import itertools
import json
import math
import os
import pickle
import random
import re
import sys
import textwrap
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path.cwd()

if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = PROJECT_ROOT / "src"
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from paths import DATA_INTERMEDIATE, DATA_RAW, FIGURES, MODELS, PROJECT_ROOT, TABLES

import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import pyfolio as pf
import seaborn as sns
import sklearn
import xgboost
import xgboost as xgb
from IPython.display import HTML, display
from scipy.optimize import minimize
from scipy.stats import entropy, norm, t
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
""",
    3: """
# Model setup, repo-relative paths, and light notebook guardrails
model = "regime_model"
model_round = "round_1"

RAW_EVENT_FILES = [
    "BoMoS_v2_1min_a 02-24 20_41_30_010220_022026.csv",
    "BoMoS_v2_1min_b 02-25 16_13_11_010213_123119.csv",
]
MASTER_DATA_FILENAME = "master_variable_inventory 20251223.xlsx"
LIQUIDITY_FILENAME = "liquidity_indices_updated_20260309.xlsx"
FEAR_GREED_FILENAME = "fear_and_greed_full.csv"

model_input_path = DATA_RAW / "xls" / "input" / model
model_round_input_path = model_input_path / model_round
model_intermediate_path = DATA_INTERMEDIATE / model / model_round
model_table_output_path = TABLES / model / model_round
model_csv_output_path = model_table_output_path / "csv"
model_xls_output_path = model_table_output_path / "xls"
model_figure_output_path = FIGURES / model / model_round
model_ml_output_path = MODELS / model / model_round
research_liquidity_path = DATA_RAW / "research" / "liquidity" / "xls"
research_fear_greed_path = DATA_RAW / "research" / "fear_greed" / "csv"

for directory in [
    model_intermediate_path,
    model_table_output_path,
    model_csv_output_path,
    model_xls_output_path,
    model_figure_output_path,
    model_ml_output_path,
]:
    directory.mkdir(parents=True, exist_ok=True)


def assert_file_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def ensure_columns(df: pd.DataFrame, required_columns, df_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"{df_name} is missing required columns: {missing_columns}")


def ensure_non_empty(df: pd.DataFrame, df_name: str) -> None:
    if df.empty:
        raise ValueError(f"{df_name} is empty. Check the upstream import and filters.")


import notebook_utils._formatting_functions as _formatting_functions
importlib.reload(_formatting_functions)
from notebook_utils._formatting_functions import (
    enrich_data_dictionary,
    event_import,
    generate_data_dictionary,
)
""",
    5: """
# Imports column names (ie. variables) and creates a dictionary: saved as "pre dictionary"
# using just one file as all the input data files have the same structure

in_file = RAW_EVENT_FILES[0]
assert_file_exists(model_round_input_path / in_file, "raw event data file")

dictionary_df = generate_data_dictionary(
    input_filename=in_file,
    input_folder=model_round_input_path,
    output_folder=model_intermediate_path,
)

if dictionary_df is None or dictionary_df.empty:
    raise ValueError("Data dictionary generation failed or returned an empty dataframe.")
""",
    6: """
# Imports master data dictionary, to assign additional categorical tags for processing: saved as "post dictionary"

master_path = assert_file_exists(
    model_round_input_path / MASTER_DATA_FILENAME,
    "master dictionary",
)

final_df = enrich_data_dictionary(
    current_dictionary_df=dictionary_df,
    master_inventory_path=master_path,
    output_folder=model_intermediate_path,
)

if final_df is None or final_df.empty:
    raise ValueError("Enriched data dictionary is empty. Review the master inventory mapping.")
""",
    8: """
# Data columns by groups: front(most important), kite (always the same), sym_ta_cols (ta colums for symbol), other symbol ta (extra_sym_cols), add-ons (extra), etc..
# 8 Lists of columns are created: front_cols, kite_cols, sym_ta_cols, extra_sym_cols, sec_cols, extra_cols, drop_cols, last_cols
front_cols = None
kite_cols = None
sym_ta_cols = None
extra_sym_cols = None
extra_cols = None
sec_cols = None
last_cols = None
drop_cols = None

ensure_columns(final_df, ["code", "kite name"], "final_df")

grouped = final_df.groupby("code")["kite name"].apply(list)
list_dict = grouped.to_dict()

for key, value in list_dict.items():
    globals()[key] = value

required_group_lists = [
    "front_cols",
    "kite_cols",
    "sym_ta_cols",
    "extra_sym_cols",
    "drop_cols",
]
missing_group_lists = [name for name in required_group_lists if globals().get(name) is None]
if missing_group_lists:
    raise ValueError(f"Missing expected column groups from master dictionary: {missing_group_lists}")

for name in required_group_lists + ["last_cols"]:
    if globals().get(name) is None:
        globals()[name] = []

front_cols = list(front_cols)
front_cols.sort(reverse=True)

print(grouped)
""",
    9: """
# MAIN RAW TRADE DATA IMPORT
for filename in RAW_EVENT_FILES:
    assert_file_exists(model_round_input_path / filename, "raw event data file")

event_frames = []

for filename in RAW_EVENT_FILES:
    new_data = event_import(
        model_round_input_path,
        filename,
        front_cols,
        kite_cols,
        sym_ta_cols,
        extra_sym_cols,
        extra_cols,
        sec_cols,
        None,
        drop_cols,
    )
    ensure_columns(new_data, ["normed_date", "entry_time", "exit_time", "symbol"], filename)
    event_frames.append(new_data)

event_data = pd.concat(event_frames, ignore_index=False)
ensure_non_empty(event_data, "event_data")
event_data.sort_values(by=["entry_time", "symbol"], ascending=[False, True], inplace=True)
""",
    14: """
DEBUG_ROWS = 1000  # None = full dataset, used only for notebook refactor/testing
outname = f"event_data {datetime.now().strftime('%Y%m%d')}.csv"
event_preview = event_data if DEBUG_ROWS is None else event_data.head(DEBUG_ROWS)
event_preview.to_csv(model_intermediate_path / outname, index=False)
print(f"Saved {len(event_preview):,} rows to {model_intermediate_path / outname}")
""",
    59: """
# Importing master list of columns/features

vars_name = LIQUIDITY_FILENAME
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

liquidity_path = assert_file_exists(research_liquidity_path / vars_name, "liquidity research file")
liquidity_df = pd.read_excel(liquidity_path, sheet_name="Sheet1", usecols=cols, parse_dates=["DATE"])

liquidity_df = liquidity_df.dropna(how="any")
liquidity_df["normed_date"] = liquidity_df["DATE"].dt.strftime("%Y-%m-%d")
""",
    62: """
# Importing master list of columns/features
vars_name = FEAR_GREED_FILENAME
cols = ["date", "value"]

fear_greed_path = assert_file_exists(research_fear_greed_path / vars_name, "fear and greed research file")
fear_df = pd.read_csv(fear_greed_path, usecols=cols)
fear_df = fear_df.dropna(how="any")
fear_df = fear_df.rename(columns={"date": "normed_date", "value": "fear_greed"})
""",
    67: """
outname = f"data_review_{datetime.now().strftime('%Y%m%d')}.csv"
review_path = model_intermediate_path / outname
summary[summary["missing"] == 0].to_csv(review_path, index=True)
print(f"Saved data review to {review_path}")
""",
    71: """
# Base Data for Static Optimization
# Deleting variables with missing rows (in this case 12-mo prices are missing 48 rows)

cols_to_drop = [col for col in missing_list if col in static_rvw_003.columns]
static_rvw_003.drop(columns=cols_to_drop, inplace=True)

print(f"Dropping {len(cols_to_drop)} columns with missing values")
print(textwrap.fill(", ".join(cols_to_drop), width=250))

partition_a = static_rvw_003.copy()
ensure_non_empty(partition_a, "partition_a")
ensure_columns(partition_a, ["normed_date", "wins", "mtm_pl"], "partition_a")
""",
    74: """
break_pct = 0.80

if partition_a["normed_date"].isna().any():
    raise ValueError("partition_a contains null normed_date values before the INS/OOS split.")

sample = int(round(len(partition_a) * break_pct, 0))
if sample <= 0 or sample >= len(partition_a):
    raise ValueError(f"Invalid INS/OOS split index {sample} for partition_a with {len(partition_a)} rows.")

sample_breakdate = partition_a.iloc[sample]["normed_date"]
print(sample_breakdate)

partition_ins = partition_a[partition_a["normed_date"] < sample_breakdate]
partition_oos = partition_a[partition_a["normed_date"] >= sample_breakdate]

ensure_non_empty(partition_ins, "partition_ins")
ensure_non_empty(partition_oos, "partition_oos")

print(len(partition_ins))
print(partition_ins["normed_date"].tail(2))
print("")
print(len(partition_oos))
print(partition_oos["normed_date"].head(2))
""",
    76: """
sample = int(round(len(partition_ins) * break_pct, 0))
if sample <= 0 or sample >= len(partition_ins):
    raise ValueError(f"Invalid INS 80/20 split index {sample} for partition_ins with {len(partition_ins)} rows.")

sample_breakdate = partition_ins.iloc[sample]["normed_date"]
print(sample_breakdate)

partition_ins_80 = partition_ins[partition_ins["normed_date"] < sample_breakdate]
partition_ins_20 = partition_ins[partition_ins["normed_date"] >= sample_breakdate]

ensure_non_empty(partition_ins_80, "partition_ins_80")
ensure_non_empty(partition_ins_20, "partition_ins_20")

print(len(partition_ins_80))
print(partition_ins_80["normed_date"].head(1))
print(partition_ins_80["normed_date"].tail(1))
print("")
print(len(partition_ins_20))
print(partition_ins_20["normed_date"].head(1))
print(partition_ins_20["normed_date"].tail(1))
""",
    79: """
DEBUG_ROWS = 50000  # None = full dataset

dfs = {
    "partition_ins_80": partition_ins_80,
    "partition_ins_20": partition_ins_20,
    "partition_oos": partition_oos,
    "partition_ins": partition_ins,
}

for name, df in dfs.items():
    ensure_non_empty(df, name)
    df_out = df if DEBUG_ROWS is None else df.head(DEBUG_ROWS)
    output_path = model_intermediate_path / f"{name}.parquet"

    df_out.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="zstd",
    )

    print(f"{name}: saved {len(df_out)} rows (original {len(df)}) to {output_path}")
""",
}


nb1_markdown_inserts = [
    (
        0,
        """
# Import And Features

This Phase 1 notebook copy keeps the existing research workflow intact while using repo-relative paths,
light validation checks, and cleaner top-to-bottom execution setup.
""",
    ),
]


nb2_updates = {
    2: """
# Standard libraries and notebook bootstrap

import ast
import gc
import importlib
import itertools
import json
import math
import os
import pickle
import random
import re
import sys
import textwrap
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path.cwd()

if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = PROJECT_ROOT / "src"
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

from paths import DATA_INTERMEDIATE, DATA_RAW, FIGURES, MODELS, PROJECT_ROOT, TABLES

import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import pyfolio as pf
import seaborn as sns
import sklearn
import xgboost
import xgboost as xgb
from IPython.display import HTML, display
from scipy.optimize import minimize
from scipy.stats import entropy, norm, t
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
""",
    5: """
# Model setup, repo-relative paths, and light notebook guardrails
model = "regime_model"
model_round = "round_1"

model_input_path = DATA_RAW / "xls" / "input" / model
model_round_input_path = model_input_path / model_round
model_intermediate_path = DATA_INTERMEDIATE / model / model_round
model_table_output_path = TABLES / model / model_round
model_csv_output_path = model_table_output_path / "csv"
model_xls_output_path = model_table_output_path / "xls"
model_figure_output_path = FIGURES / model / model_round
model_ml_output_path = MODELS / model / model_round
research_liquidity_path = DATA_RAW / "research" / "liquidity" / "xls"
research_fear_greed_path = DATA_RAW / "research" / "fear_greed" / "csv"

for directory in [
    model_intermediate_path,
    model_table_output_path,
    model_csv_output_path,
    model_xls_output_path,
    model_figure_output_path,
    model_ml_output_path,
]:
    directory.mkdir(parents=True, exist_ok=True)


def assert_file_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def ensure_columns(df: pd.DataFrame, required_columns, df_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"{df_name} is missing required columns: {missing_columns}")


def ensure_non_empty(df: pd.DataFrame, df_name: str) -> None:
    if df.empty:
        raise ValueError(f"{df_name} is empty. Check the upstream parquet export.")
""",
    9: """
names = [
    "partition_ins_80",
    "partition_ins_20",
    "partition_oos",
    "partition_ins",
]

for name in names:
    assert_file_exists(model_intermediate_path / f"{name}.parquet", f"{name} parquet")

loaded = {
    name: pd.read_parquet(model_intermediate_path / f"{name}.parquet", engine="pyarrow")
    for name in names
}

partition_ins_80 = loaded["partition_ins_80"]
partition_ins_20 = loaded["partition_ins_20"]
partition_oos = loaded["partition_oos"]
partition_ins = loaded["partition_ins"]

required_loaded_columns = ["normed_date", "wins", "mtm_pl"]
for name, df in loaded.items():
    ensure_non_empty(df, name)
    ensure_columns(df, required_loaded_columns, name)

print({k: v.shape for k, v in loaded.items()})
""",
    10: """
feature_columns = build_feature_list_from_columns(partition_ins_80)
if not feature_columns:
    raise ValueError("No feature columns were selected from partition_ins_80.")

missing_feature_columns = [col for col in feature_columns if col not in partition_ins_20.columns or col not in partition_oos.columns]
if missing_feature_columns:
    raise KeyError(f"Selected features missing from downstream partitions: {missing_feature_columns}")

print(f"variables:{len(feature_columns)}")
print(textwrap.fill(", ".join(feature_columns), width=250))
""",
    40: """
import notebook_utils._formatting_functions as _formatting_functions
importlib.reload(_formatting_functions)
from notebook_utils._formatting_functions import nan_inf_summary

summary_df = nan_inf_summary(partition_ins_80, df_name="partition_ins_80")
""",
    42: """
# Two options: all_features or selected_features

USE_SELECTED_FEATURES = True
ml_list = selected_features if USE_SELECTED_FEATURES else all_features
ml_list = list(dict.fromkeys(ml_list))

if not ml_list:
    raise ValueError("ml_list is empty. Review the feature selection section before training.")

print(len(ml_list))
""",
    43: """
for df_name, df in {
    "partition_ins_80": partition_ins_80,
    "partition_ins_20": partition_ins_20,
    "partition_oos": partition_oos,
}.items():
    ensure_columns(df, ml_list + ["wins"], df_name)

X_train = partition_ins_80[ml_list]
y_train = partition_ins_80["wins"]

X_test = partition_ins_20[ml_list]
y_test = partition_ins_20["wins"]

# Only dependent variable needed for OOS as we will fit the optimal number of features of X, which is calculated later
y_oos = partition_oos["wins"]

print(f"train: {len(y_train)}")
print("Train Win %", np.mean(y_train))
print("")
print(f"test: {len(y_test)}")
print("Test Win %", np.mean(y_test))
print("")
print(f"oos: {len(y_oos)}")
print("OOS Win %", np.mean(y_oos))
""",
    50: """
best_params = grid_search.best_params_

# Step 1: Create a new XGB with the best parameters
best_xgb = xgb.XGBClassifier(
    **best_params,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    objective="binary:logistic",
    use_label_encoder=False,
    tree_method="hist",
    n_jobs=-1,
)

# Step 2: Train the model on your training data
best_xgb.fit(X_train, y_train)

# Step 3: Evaluate or use the model
pred_train = best_xgb.predict(X_train)
pred_test = best_xgb.predict(X_test)

print(classification_report(y_train, pred_train))
print(confusion_matrix(y_train, pred_train))

print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
""",
    53: """
X_train_indexed = partition_ins_80.copy()
X_train_indexed["normed_date"] = pd.to_datetime(X_train_indexed["normed_date"])
X_train_indexed = X_train_indexed.set_index("normed_date")

daily_counts = X_train_indexed.resample("1D").size()

# Filter out days with 0 trades (weekends/holidays) to get a true average
active_days = daily_counts[daily_counts > 0]
if active_days.empty:
    raise ValueError("No active trading days found in partition_ins_80 after indexing by normed_date.")

avg_rows_per_day = active_days.mean()
median_rows_per_day = active_days.median()

print(f"Average rows per day: {int(avg_rows_per_day)}")
print(f"Median rows per day:  {int(median_rows_per_day)}")

# 2. Define your desired "Window" in time
# For event-driven strategies, 3-6 months is common for regime adaptation.
target_lookback_months = 9
target_lookback_days = target_lookback_months * 21  # ~21 trading days/month

# 3. Calculate max_train_size
calculated_size = int(avg_rows_per_day * target_lookback_days)
if calculated_size <= 0:
    raise ValueError(f"Calculated max_train_size must be positive, got {calculated_size}.")

print(f"Recommended max_train_size: {calculated_size}")
""",
    54: """
MAX_TRAIN_SIZE = calculated_size
cv_rolling = TimeSeriesSplit(max_train_size=MAX_TRAIN_SIZE, n_splits=5)

# Setting finer steps (to 1 from 10)
rfecv = RFECV(
    estimator=best_xgb,
    step=1,
    min_features_to_select=5,
    cv=cv_rolling,
    scoring="precision",
    n_jobs=-1,
)

rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")

selected_mask = rfecv.support_
selected_features = X_train.columns[selected_mask]
if len(selected_features) == 0:
    raise ValueError("RFECV did not return any selected features.")

print(f"Optimal features: {list(selected_features)}")
""",
    60: """
rfecv_vars = selected_features.tolist()
if not rfecv_vars:
    raise ValueError("rfecv_vars is empty. Review the RFECV results before continuing.")

for df_name, df in {
    "partition_ins_80": partition_ins_80,
    "partition_ins_20": partition_ins_20,
    "partition_oos": partition_oos,
}.items():
    ensure_columns(df, rfecv_vars, df_name)

X_train_rfecv = partition_ins_80[rfecv_vars]
X_test_rfecv = partition_ins_20[rfecv_vars]
X_oos_rfecv = partition_oos[rfecv_vars]
""",
    61: """
# Fitting/evaluating: INS Train
best_xgb.fit(X_train_rfecv, y_train)

# Predicting/evaluating: INS Train
pred_train_rfecv = best_xgb.predict(X_train_rfecv)
yhat_train = best_xgb.predict_proba(X_train_rfecv)
yhat_train_1 = yhat_train[:, 1]
if len(yhat_train_1) != len(X_train_rfecv):
    raise ValueError("Train prediction length does not match X_train_rfecv.")
print("Obs:", len(yhat_train_1))
print("Mean:", np.mean(yhat_train_1))

# Predicting/evaluating: INS Test
pred_test_rfecv = best_xgb.predict(X_test_rfecv)
yhat_test = best_xgb.predict_proba(X_test_rfecv)
yhat_test_1 = yhat_test[:, 1]
if len(yhat_test_1) != len(X_test_rfecv):
    raise ValueError("Test prediction length does not match X_test_rfecv.")
print("Obs:", len(yhat_test_1))
print("Mean:", np.mean(yhat_test_1))

# Predicting/evaluating: OOS
pred_oos_rfecv = best_xgb.predict(X_oos_rfecv)
yhat_oos = best_xgb.predict_proba(X_oos_rfecv)
yhat_oos_1 = yhat_oos[:, 1]
if len(yhat_oos_1) != len(X_oos_rfecv):
    raise ValueError("OOS prediction length does not match X_oos_rfecv.")
print("Obs:", len(yhat_oos_1))
print("Mean:", np.mean(yhat_oos_1))
""",
    66: """
model = "regime_model"
model_round = "round_1"
direction = "long" if LONG_SHORT == 1 else "short"
filename = f"xgb_{model}_{direction}.pkl"

model_ml_output_path.mkdir(parents=True, exist_ok=True)
ml_path = model_ml_output_path / filename

with open(ml_path, "wb") as f:
    pickle.dump(best_xgb, f)

print(f"Model saved at: {ml_path}")
print(f"RFECV feature count: {len(rfecv_vars)}")
""",
    71: """
ML_PROBA_COL = "ml_proba_1"

prediction_map = {
    "partition_ins_80_001": (partition_ins_80_001, yhat_train_1),
    "partition_ins_20_001": (partition_ins_20_001, yhat_test_1),
    "partition_oos_001": (partition_oos_001, yhat_oos_1),
}

for df_name, (df, yhat) in prediction_map.items():
    if len(df) != len(yhat):
        raise ValueError(f"Length mismatch between {df_name} and prediction array.")

    df[ML_PROBA_COL] = yhat
    df.reset_index(drop=True, inplace=True)
    ensure_columns(df, [ML_PROBA_COL], df_name)
""",
    77: """
DEBUG_ROWS = 50000  # or 50000 for quick iteration

dfs = {
    "partition_ins_80_001": partition_ins_80_001,
    "partition_ins_20_001": partition_ins_20_001,
    "partition_oos_001": partition_oos_001,
}

for name, df in dfs.items():
    ensure_non_empty(df, name)
    df_out = df if DEBUG_ROWS is None else df.head(DEBUG_ROWS)
    output_path = model_intermediate_path / f"{name}.parquet"

    print(f"{name}: writing {len(df_out):,} rows to {output_path}")
    df_out.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression="zstd",
    )
""",
}


nb2_markdown_inserts = [
    (
        0,
        """
# XGB Optimization

This Phase 1 notebook copy keeps the current optimization workflow intact while making path handling,
artifact loading, and prediction exports safer to rerun from top to bottom.
""",
    ),
]


if __name__ == "__main__":
    write_notebook(
        "1 - bomo v2 QQQ long ML backtest - import & features v00.ipynb",
        "1 - bomo v2 QQQ long ML backtest - import & features v01.ipynb",
        nb1_updates,
        markdown_inserts=nb1_markdown_inserts,
    )
    write_notebook(
        "2 - bomo v2 QQQ long ML backtest - XGB optimization v00.ipynb",
        "2 - bomo v2 QQQ long ML backtest - XGB optimization v01.ipynb",
        nb2_updates,
        markdown_inserts=nb2_markdown_inserts,
    )
