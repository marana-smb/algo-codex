import numpy as np
import pandas as pd
import operator
import math
import itertools
from datetime import datetime, date
import pyfolio as pf
from scipy.stats import t
from scipy.optimize import minimize
import ast
import matplotlib.pyplot as plt

# 3.7.16 specific
from typing import List

# 0.4 OPTIMIZATION FUNCTIONS

# Main variables dictionary
sum_cols = ['mtm_pl', 'entry_pl', 'matched_shares', 'entry_side', 'entry_fees', 'exit_fees', 'exit_shares', 'pl_g', 'pl_n', 'fees']
sum_dict = {key: 'sum' for key in (sum_cols)}

############################################################################################################################################################################################
# Func 11A: Filters data applying criteria
# def optmz_search(dta_in, criteria: dict):
#     """
#     Apply filters to a DataFrame based on criteria.
#     dta_in (pd.DataFrame): Input DataFrame to filter.
#     criteria (dict): Dictionary containing column names as keys and tuples of (value, operator) as values.
#     Returns: pd.DataFrame: Filtered DataFrame.
#     """
#     mask = pd.Series([True] * len(dta_in))

#     for key, (value, operator) in criteria.items():
#         if not key in dta_in.columns:
#             raise ValueError(f"Column '{key}' not found in DataFrame")
        
#         if not pd.api.types.is_numeric_dtype(dta_in[key]):
#             raise ValueError(f"Column '{key}' does not contain numeric values")

#         if operator == '>=':
#             mask = mask & (dta_in[key] >= value)
#         elif operator == '<=':
#             mask = mask & (dta_in[key] <= value)
#         elif operator == '>':
#             mask = mask & (dta_in[key] > value)
#         elif operator == '<':
#             mask = mask & (dta_in[key] < value)
#         elif operator == '==':
#             mask = mask & (dta_in[key] == value)
#         else:
#             raise ValueError(f"Unsupported condition: {operator}")
        
#         filtered_df = dta_in[mask]
#         obs = filtered_df.shape[0]
#         tobs = len(dta_in)

#         sum_cols = ['matched_shares', 'pl_g', 'pl_n', 'fees']
#         sum_dict = {key: 'sum' for key in (sum_cols)}
#         pnl = filtered_df.agg(sum_dict).reset_index()
#         shares = pnl.iloc[0,1]
#         pl_g = pnl.iloc[1,1]
#         pl_n = pnl.iloc[2,1]
#         fees = pnl.iloc[3,1]

#     return filtered_df, obs, tobs, shares, pl_g, pl_n, fees

def optmz_search(dta_in, criteria: dict, min_obs: int = 25):
    """
    Apply filters to a DataFrame based on criteria.
    
    Parameters:
        dta_in (pd.DataFrame): Input DataFrame to filter.
        criteria (dict): Column names mapped to (value, operator) tuples.
        min_obs (int): Minimum number of rows required to return filtered results.
        
    Returns:
        tuple: (filtered_df or None, obs, tobs, shares, pl_g, pl_n, fees)
    """
    mask = pd.Series([True] * len(dta_in))

    for key, (value, operator) in criteria.items():
        if key not in dta_in.columns:
            raise ValueError(f"Column '{key}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(dta_in[key]):
            raise ValueError(f"Column '{key}' does not contain numeric values")

        if operator == '>=':
            mask = mask & (dta_in[key] >= value)
        elif operator == '<=':
            mask = mask & (dta_in[key] <= value)
        elif operator == '>':
            mask = mask & (dta_in[key] > value)
        elif operator == '<':
            mask = mask & (dta_in[key] < value)
        elif operator == '==':
            mask = mask & (dta_in[key] == value)
        else:
            raise ValueError(f"Unsupported condition: {operator}")

    filtered_df = dta_in[mask]
    obs = filtered_df.shape[0]
    tobs = len(dta_in)

    if obs < min_obs:
        return None, obs, tobs, 0.0, 0.0, 0.0, 0.0

    sum_cols = ['matched_shares', 'pl_g', 'pl_n', 'fees']
    sum_dict = {key: 'sum' for key in sum_cols}
    pnl = filtered_df.agg(sum_dict).reset_index()
    shares = pnl.iloc[0, 1]
    pl_g = pnl.iloc[1, 1]
    pl_n = pnl.iloc[2, 1]
    fees = pnl.iloc[3, 1]

    return filtered_df, obs, tobs, shares, pl_g, pl_n, fees

# Func 11B: Includes excluded rows and explanations
def optmz_search_with_exclusions(dta_in, criteria: dict, min_obs: int = 25):
    mask = pd.Series(True, index=dta_in.index)
    failed_reasons = pd.DataFrame(index=dta_in.index)

    for key, (value, operator) in criteria.items():
        if operator == '>=':
            pass_mask = dta_in[key] >= value
        elif operator == '<=':
            pass_mask = dta_in[key] <= value
        elif operator == '>':
            pass_mask = dta_in[key] > value
        elif operator == '<':
            pass_mask = dta_in[key] < value
        elif operator == '==':
            pass_mask = dta_in[key] == value
        else:
            raise ValueError(f"Unsupported condition: {operator}")

        # Identify failing rows
        failed_rows = ~pass_mask
        reason_str = f"{key} {operator} {value}"
        failed_reasons.loc[failed_rows, key] = reason_str

        # Combine with main mask
        mask &= pass_mask

    filtered_df = dta_in[mask]
    excluded_df = dta_in[~mask].copy()
    excluded_df['fail_reason'] = failed_reasons.apply(lambda row: '; '.join(row.dropna()), axis=1)

    obs = filtered_df.shape[0]
    tobs = len(dta_in)

    if obs < min_obs:
        return None, obs, tobs, 0.0, 0.0, 0.0, 0.0, excluded_df

    sum_cols = ['matched_shares', 'pl_g', 'pl_n', 'fees']
    sum_dict = {key: 'sum' for key in sum_cols}
    pnl = filtered_df.agg(sum_dict).reset_index()
    shares = pnl.iloc[0, 1]
    pl_g = pnl.iloc[1, 1]
    pl_n = pnl.iloc[2, 1]
    fees = pnl.iloc[3, 1]

    return filtered_df, obs, tobs, shares, pl_g, pl_n, fees, excluded_df

############################################################################################################################################################################################
# COMPANION to #18. Func 12: Creating data by Date for time series curves/analysis
def create_bydate_data(dta_in: pd.DataFrame, sum_dict: dict, bp: int) -> pd.DataFrame:
    """
    Collapse data to create time series
    dta_in (pd.DataFrame): Input DataFrame to filter.
    criteria (dict): Dictionary containing column names as keys and operation (sum) as value.
    Returns: pd.DataFrame
    """
    dta_out = dta_in.groupby(['normed_date']).agg(sum_dict).reset_index()
    dta_out['cum_pl_g'] = dta_out['pl_g'].cumsum()
    dta_out['cum_pl_n'] = dta_out['pl_n'].cumsum()
    dta_out['high_cum_pl_n'] = dta_out['cum_pl_n'].cummax()
    dta_out['low_pl_n'] = dta_out['pl_n'].cummin()
    dta_out['high_pl_n'] = dta_out['pl_n'].cummax()
    dta_out['wins'] = (dta_out['pl_n'] > 0).astype(int)
    dta_out['share_pl_n'] = dta_out['pl_n'] / dta_out['matched_shares']
    dta_out['ret_g'] = dta_out['pl_g'] / bp
    dta_out['ret_n'] = dta_out['pl_n'] / bp
    dta_out['cum_ret_g'] = 1 + dta_out['ret_g']
    dta_out['cum_ret_n'] = 1 + dta_out['ret_n']
    dta_out['comp_ret_g'] = dta_out['cum_ret_g'].cumprod()
    dta_out['comp_ret_n'] = dta_out['cum_ret_n'].cumprod()
    dta_out['high_comp_ret_n'] = dta_out['cum_ret_n'].cummax()
    return dta_out

############################################################################################################################################################################################
# Func 13: Calculates performance stats 
#def create_performance(dta_in, var: str, id: int, obs: int):
def create_performance(dta_in, var: str, id: int, obs: int, tobs: int, shares: float, pl_g: float, pl_n: float, fees: float, crit: dict):
    """
    Calculate performance metrics
    dta_in (pd.DataFrame): Input DataFrame to filter.
    var: return variable.
    id: identifier of the run
    obs: number of observations
    shares: total matched shares
    pl_g: gross profit/loss
    pl_n: net profit/loss
    fees: total fees
    crit: criteria used for filtering (to be included in the output)
    Returns: pd.DataFrame
    """
    
    ret_arr = np.array(dta_in[var])
    tdays = len(ret_arr)
    per_stats = pf.timeseries.perf_stats(ret_arr)
    per_stats_df = pd.DataFrame(per_stats, columns=['Value']).T.reset_index()
    # Addition
    per_stats_df['ui'] = ulcer_index(dta_in)
    per_stats_df['upi'] = (per_stats_df['Cumulative returns'] / per_stats_df['ui']).round(5) * 100
    per_stats_df['index'] = id
    per_stats_df['obs'] = obs
    per_stats_df['data_pct'] = round(obs / tobs, 3)
    per_stats_df['shares'] = shares
    per_stats_df['pl_n_share'] = (pl_n / per_stats_df['shares']).round(3)
    per_stats_df['pl_g'] = round(pl_g, 0)
    per_stats_df['pl_n'] = round(pl_n, 0)
    per_stats_df['fees'] = round(fees, 0)
    per_stats_df['trade_days'] = tdays
    crit_str = str(crit)
    per_stats_df['criteria'] = crit_str
    return per_stats_df

############################################################################################################################################################################################
# Func 14: Calculates P-values - source: the statistics of Sharpe Ratios (Lo)
def sr_ttest(dta_in):
    """ Calculates p-values for Sharpe Ratios.
        Usage: sr_ttest(final_results(5)) 
    """
    
    # Calculate the standard error of the Sharpe Ratio
    dta_in['sr_se'] = (np.sqrt((1 + (dta_in['Sharpe ratio']**2 / 2)) / dta_in['obs'])).round(3)
    
    # Calculate the t-statistic
    dta_in['t_stat'] = (dta_in['Sharpe ratio'] / dta_in['sr_se']).round(3)
    
    # Calculate the p-value
    dta_in['p_value'] = (2 * (1 - t.cdf(abs(dta_in['t_stat']), df = dta_in['obs']-1))).round(4)
    
    return dta_in

# COMPANION #15 Func 7D: Decile summary
def decile_summary(dta_in, var: str):
    """ Produces quick summary with deciles
        usage: decile_summary(stat_optmz_sample, 'dist_entry_to_lod')
    """
    summary = dta_in[var].describe()
    deciles = dta_in[var].quantile([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    combined_summary = pd.concat([summary, deciles])
    return combined_summary

############################################################################################################################################################################################
# Func 15: Calculates ranges and buckets for optimization criteria
def optimization_ranges(dta_in: pd.DataFrame, optmz_list: list, drop_manual: list, stps: int = 20):
    """
    This function calculates optimization ranges based on a set of decile summaries and produces a cleaned output DataFrame with steps for optimization.
    - dta_in: Input DataFrame containing the data to summarize.
    - optmz_list: List of variables to be optimized.
    - dta_out: Output DataFrame where optimization ranges and steps will be stored.
    - drop_manual: List of column names to be dropped manually.
    - stps: Number of steps to be used for optimization.
    """
    # Initialize an empty DataFrame to hold the summaries for each variable
    df = pd.DataFrame()

    # Loop through the optimization list and compute decile summaries
    for var in optmz_list:
        summary = decile_summary(dta_in, var)
        df[var] = summary  # Store the summary in the DataFrame with the variable name as the column
    
    # Clean and transform data
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    # df2 = df.dropna(axis=1, how='any')
    df3 = df[(df.index == 0.1) | (df.index == 0.9)]
    df4 = df3.drop(columns=drop_manual)
    df5 = np.floor(df4 * 1000) / 1000
    df_trsp = df5.T

    # Calculate step sizes for each variable and add to the output DataFrame
    df_trsp['steps'] = (df_trsp[0.9] - df_trsp[0.1]) / stps

    # Return the modified output DataFrame
    return df_trsp, df

############################################################################################################################################################################################
# Func 16: Calculate ulcer index
def ulcer_index(dta_in: pd.DataFrame) -> float:
    """
    Calculate the Ulcer Index from the input DataFrame.    
    dta_in (pd.DataFrame): DataFrame containing 'comp_ret_n' and 'high_comp_ret_n' columns.
    Returns float: The Ulcer Index calculated from the squared drawdown.
    """
    
    # Ensure the required columns are present
    if 'comp_ret_n' not in dta_in.columns or 'high_comp_ret_n' not in dta_in.columns:
        raise ValueError("DataFrame must contain 'comp_ret_n' and 'high_comp_ret_n' columns.")
    
    # Calculate drawdown and squared drawdown
    dta_in['draw_down'] = ((dta_in['comp_ret_n'] - dta_in['high_comp_ret_n']) / dta_in['high_comp_ret_n']) * 100
    dta_in['sq_draw_down'] = dta_in['draw_down'] ** 2
    
    # Compute the Ulcer Index
    # dta_in['ui'] = np.sqrt(dta_in['sq_draw_down'].mean()).round(3)
    ui = np.sqrt(dta_in['sq_draw_down'].mean()).round(3)
    return ui

############################################################################################################################################################################################
# Func 17: Process min/max run for two variables
def process_optmz_minmax(dta_in: pd.DataFrame, target_a: str, max_id: int=-1) -> pd.DataFrame:
# def process_optmz_minmax(dta_in: pd.DataFrame, target_a: str, target_b: str,max_id: int=-1) -> pd.DataFrame:
    """
    Processes optimization results and returns a DataFrame with the top and bottom results
    for the specified targets (target_a and target_b).
        dta_in (pd.DataFrame): Input DataFrame containing the data.
        target_a (str): Column name for the first target to sort by.
        target_b (str): Column name for the second target to sort by.
        pd.DataFrame: A DataFrame with the concatenated top and bottom results for each variable.
    """
    
    # Copy the input DataFrame to avoid modifying the original data
    df = dta_in.copy()
    
    # Extract variable names from the 'criteria' column
    df['variable'] = df['criteria'].str.extract(r"\{'(\w+)'")
    

    # Sort by target_a and get the top and bottom values for each variable
    df_a = df.sort_values(by=['variable', target_a], ascending=[True, False])
    df_a1 = df_a.groupby('variable').apply(lambda x: x.iloc[[0, max_id]]).reset_index(drop=True)
    
    # Sort by target_b and get the top and bottom values for each variable
    # df_b = df.sort_values(by=['variable', target_b], ascending=[True, False])
    # df_b1 = df_b.groupby('variable').apply(lambda x: x.iloc[[0, max_id]]).reset_index(drop=True)
    
    # Concatenate the two DataFrames and remove duplicates
    # df_ab1 = pd.concat([df_a1, df_b1], ignore_index=True)
    # df_ab1 = df_ab1.sort_values(by=['variable', target_b], ascending=[True, False])
    # df_ab2 = df_ab1.drop_duplicates()
    
    # return df_ab2
    return df_a1, df_a

############################################################################################################################################################################################
# Func 18A: Wraps the optimization/search loop of 3 functions
# def optmz_loop_wrap(dta_in: pd.DataFrame, crit_lst: list, bp) -> pd.DataFrame:
#     results = []
#     for i, crit in enumerate(crit_lst, start=1):
#         try:
#             optmz, obs, tobs, shares, pl_g, pl_n, fees = optmz_search(dta_in, crit)
#             optmz_bydate = create_bydate_data(optmz, sum_dict, bp)
#             optmz_perf = create_performance(optmz_bydate, 'ret_n', i, obs, tobs, shares, pl_g, pl_n, fees, crit)
#             results.append(optmz_perf)
#         except Exception as e:
#             print(f"Error processing criterion {crit}: {e}")
    
#     dta_out = pd.concat(results, ignore_index=True)
#     return dta_out, optmz, optmz_bydate

def optmz_loop_wrap(dta_in: pd.DataFrame, crit_lst: list, bp) -> pd.DataFrame:
    results = []
    for i, crit in enumerate(crit_lst, start=1):
        try:
            optmz, obs, tobs, shares, pl_g, pl_n, fees = optmz_search(dta_in, crit)

            # Skip if not enough observations
            if optmz is None:
                print(f"Skipping criterion {crit}: not enough observations.")
                continue

            optmz_bydate = create_bydate_data(optmz, sum_dict, bp)
            optmz_perf = create_performance(optmz_bydate, 'ret_n', i, obs, tobs, shares, pl_g, pl_n, fees, crit)
            results.append(optmz_perf)

        except Exception as e:
            #print(f"Error processing criterion {crit}: {e}")
            pass
    
    # Only concat if we got results
    dta_out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return dta_out, optmz, optmz_bydate if results else (None, None)

# Func 18B: Wraps the optimization/search loop of 3 functions
def optmz_loop_wrap_with_exclusions(dta_in: pd.DataFrame, crit_lst: list, bp) -> tuple:
    results = []
    all_excluded = []
    optmz = None
    optmz_bydate = None

    for i, crit in enumerate(crit_lst, start=1):
        try:
            optmz, obs, tobs, shares, pl_g, pl_n, fees, excluded = optmz_search_with_exclusions(dta_in, crit)

            if optmz is None:
                print(f"Skipping criterion {crit}: not enough observations.")
                all_excluded.append(excluded)
                continue

            sum_dict = {'matched_shares': 'sum', 'pl_g': 'sum', 'pl_n': 'sum', 'fees': 'sum'}
            optmz_bydate = create_bydate_data(optmz, sum_dict, bp)
            optmz_perf = create_performance(optmz_bydate, 'ret_n', i, obs, tobs, shares, pl_g, pl_n, fees, crit)
            results.append(optmz_perf)
            all_excluded.append(excluded)

        except Exception as e:
            print(f"Error processing criterion {crit}: {e}")
            continue

    dta_out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    excluded_all = pd.concat(all_excluded, ignore_index=True) if all_excluded else pd.DataFrame()

    return dta_out, optmz, optmz_bydate, excluded_all

############################################################################################################################################################################################
# Func 19: Finds break points in the optimization variable (var_name) to analyze down the road
def process_optmz_breaks(dta_in: pd.DataFrame, var_group: str, var_name: str, low: float, high: float) -> pd.DataFrame:
    # Finds the first and last rows where 'var_name' is between 'low' and 'high' within each group
    def find_transition(group, var_name: str, low: float, high: float):
        transition_rows = group[(group[var_name] > low) & (group[var_name] < high)]
        return pd.concat([transition_rows.head(1), transition_rows.tail(1)])
    
    # Apply the transition logic for each group in 'var_group'
    dta_out = dta_in.groupby(var_group, group_keys=False).apply(find_transition, var_name=var_name, low=low, high=high)
    return dta_out

############################################################################################################################################################################################
# Func 20: calculate total number of combinations
# A
def comb(n: int, k: int) -> int:
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
# B
# def sum_combinations(n: int, x: int) -> int:
#     total_combinations = sum(math.comb(n, k) for k in range(1, x+1))
#     return total_combinations

############################################################################################################################################################################################
def sum_combinations(n: int, x: int) -> int:
    total_combinations = sum(comb(n, k) for k in range(1, x + 1))
    return total_combinations

############################################################################################################################################################################################
# Func 21: Calculate combinations of optimization criteria
def create_combinations(dim_lst: List[str], in_lst: List[str], cmax: int = 6) -> List[str]:
    all_combinations = []
    
    # Create combinations up to the specified max count of elements per combination
    for r in range(1, min(len(dim_lst), cmax+1)):
        combinations = list(itertools.combinations(in_lst, r))
        all_combinations.extend(combinations)
    
    # Convert each combination tuple to a string format
    combination_strings = [', '.join(combo) for combo in all_combinations]
    
    # Create a DataFrame from the combinations
    combinations_df = pd.DataFrame({'combinations': combination_strings})
    
    # Convert DataFrame to list
    out_lst = combinations_df['combinations'].tolist()
    print(f"Number of combinations: {len(out_lst)}")
    return out_lst

############################################################################################################################################################################################
# Func 22: Converts text string to proper dictionary format
def text_to_dict(in_lst: List[str]) -> List[dict]:
    """
    Converts a list of string representations of dictionaries or tuples of dictionaries
    into a list of combined dictionaries.
    """
    list_of_dicts = []
    
    for item in in_lst:
        try:
            evaluated_item = ast.literal_eval(item)
        except (ValueError, SyntaxError) as e:
            print(f"Skipping invalid entry: {item}. Error: {e}")
            continue
        
        combined_dict = {}
        
        if isinstance(evaluated_item, dict):
            combined_dict.update(evaluated_item)
        elif isinstance(evaluated_item, tuple):
            for d in evaluated_item:
                if isinstance(d, dict):
                    combined_dict.update(d)
                else:
                    print(f"Skipping non-dict element within tuple: {d}")
        
        list_of_dicts.append(combined_dict)
    
    return list_of_dicts

############################################################################################################################################################################################
# Func 23: Find outliers in a DataFrame column based on a quantile and a comparison operator.
def outlier_bound(dta_in: pd.DataFrame, var: str, quant: float, oper):
    """
    Find outliers in a DataFrame column based on a quantile and a comparison operator.
        dta_in (pd.DataFrame): Input DataFrame.
        var (str): Column name to analyze.
        quant (float): Quantile value (e.g., 0.01 for 1%).
        oper (callable): Comparison operator (e.g., operator.lt, operator.gt).
        Returns: DataFrame containing rows that are outliers and count of rows in df.
    Sample usage: sample, atr_cnt = outlier_bound(partition_ins_80, 'atr', 0.01, operator.lt)

    """
    # Calculate the quantile bound
    bound = dta_in[var].quantile(quant)
    
    # Apply the comparison operator to find outliers
    outliers = dta_in[oper(dta_in[var], bound)]
    print(f"{var} :{len(outliers)} obs")
    cnt = len(outliers)
    
    return outliers, cnt

############################################################################################################################################################################################
# Func 24: Create chart of features for older version of sklearn
def _infer_feature_counts_for_rfecv_020(n_total_features: int, step, min_features: int, n_scores: int) -> np.ndarray:
    """
    Infer the number of features evaluated at each RFECV iteration for sklearn 0.20.x.

    RFECV evaluates score with current feature set, then removes `step` features (or
    a fraction if step is float) until reaching `min_features_to_select`.
    """
    if n_total_features <= 0:
        raise ValueError("n_total_features must be > 0")
    min_features = max(1, int(min_features))

    counts = []
    cur = int(n_total_features)

    if isinstance(step, float):
        # Approximate sklearn's fractional step logic.
        step = float(step)
        while cur > min_features and len(counts) < n_scores:
            counts.append(cur)
            remove = max(1, int(step * cur))
            cur = max(min_features, cur - remove)
        if len(counts) < n_scores:
            counts.append(cur)
        return np.array(counts[:n_scores], dtype=int)

    # int step
    step = max(1, int(step))
    while cur > min_features and len(counts) < n_scores:
        counts.append(cur)
        cur = max(min_features, cur - step)
    if len(counts) < n_scores:
        counts.append(cur)
    return np.array(counts[:n_scores], dtype=int)

def plot_features_vs_cvscore_rfecv_020(rfecv_model, X, scoring_label: str = "recall", increasing_x: bool = True):
    """
    Compatible with sklearn 0.20.3: uses rfecv_model.grid_scores_.

    - sklearn 0.20.x RFECV does not expose cv_results_
    - grid_scores_ contains mean CV score per iteration (no std in this API)
    """
    if not hasattr(rfecv_model, "grid_scores_"):
        raise AttributeError("Expected RFECV(grid_scores_) for sklearn 0.20.x. Did you fit() the model?")

    scores = np.asarray(rfecv_model.grid_scores_, dtype=float)
    n_scores = len(scores)

    n_total = int(getattr(X, "shape")[1])
    step = getattr(rfecv_model, "step", 1)
    min_features = int(getattr(rfecv_model, "min_features_to_select", 1))

    x = _infer_feature_counts_for_rfecv_020(
        n_total_features=n_total,
        step=step,
        min_features=min_features,
        n_scores=n_scores,
    )

    # Defensive align
    m = min(len(x), len(scores))
    x, scores = x[:m], scores[:m]

    if increasing_x:
        order = np.argsort(x)
        x, scores = x[order], scores[order]

    plt.figure(figsize=(10, 5))
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Mean CV {scoring_label}")
    plt.plot(x, scores)
    plt.title(f"RFECV mean CV {scoring_label} vs number of features (sklearn 0.20.x)")
    plt.grid(True, alpha=0.2)
    plt.show()
