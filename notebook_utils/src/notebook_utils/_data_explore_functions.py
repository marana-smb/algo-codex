import numpy as np
import pandas as pd
import math
import itertools
from datetime import datetime, date
from IPython.display import display, HTML
import pyfolio as pf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import t

## 0.3 DATA EXPLORATION FUNCTIONS (CROSSTABS, CHARTS)

# Func 6: for bygroup summary
def cross_tabs(dta_in, bygroup, var):
    """ Used to create cross tabs where 'bygroup' is categorical and/or buckets have been created (days in week, year, etc..).
        bygroup: variable(s) to do the cross-tabulation against.
        var: variable of interest, usually the P/L of the strategy.
        sample usage: weekday_pl = cross_tabs(static_rvw, ['week_day'], 'pl_n')
    """  
    dta_out = dta_in.groupby(bygroup).agg({var: ['sum', 'count']}).reset_index()
    dta_out.columns = bygroup + [f'{var}_sum', f'{var}_cnt']
    dta_out[f'{var}_pct'] = dta_out[f'{var}_sum'] / dta_out[f'{var}_sum'].sum()
    return dta_out

# Func 7B: P/L cross-tabs by selected categories and date/time vectors - IMPROVED FUNCTION (no cat_var)
def explore_cross(dta_in, main_var, key_var, _cats, _labels):
    """ Used to create cross tabs by categories/bins AND P/L across calendar variables at once.
        main_var: main variable that needs new categories/bins to be analyzed.
        key_var: usually P/L.
        _cats: breakpoints on main_var.
        _labels: categories/bins created from breakpoints.
        sample usage: dist_entry_to_open_pl = explore_cross_two(static_rvw, 'dist_entry_to_open', 'pl_n', ratio_cats, ratio_labs)
    """  
    dta_in[f'{main_var}_cat'] = pd.cut(dta_in[main_var], bins = _cats, labels = _labels, right = False)
    _pl = cross_tabs(dta_in, [f'{main_var}_cat'], key_var)
    _pl_week_day = cross_tabs(dta_in, [f'{main_var}_cat', 'week_day'], key_var)
    _pl_am_pm = cross_tabs(dta_in, [f'{main_var}_cat', 'am_pm'], key_var)
    _pl_mon = cross_tabs(dta_in, [f'{main_var}_cat', 'month'], key_var)
    _pl_yr = cross_tabs(dta_in, [f'{main_var}_cat', 'year'], key_var)
    #_pl_quarter_hr = cross_tabs(dta_in, [f'{main_var}_cat', 'quarter_hr'], key_var)
    #_pl_day_qrt_hr = cross_tabs(dta_in, [f'{main_var}_cat', 'day_qrt_hr'], key_var)
    
    #dta_out = pd.concat([_pl, _pl_week_day, _pl_am_pm, _pl_mon, _pl_yr, _pl_quarter_hr, _pl_day_qrt_hr], axis = 0, ignore_index = True)
    dta_out = pd.concat([_pl, _pl_week_day, _pl_am_pm, _pl_mon, _pl_yr], axis = 0, ignore_index = True)
    
    main_cols = [f'{key_var}_sum', f'{key_var}_pct']
    dta_out = dta_out[main_cols + [col for col in dta_out.columns if col not in main_cols]]
    dta_out = dta_out[dta_out[f'{key_var}_pct'] != 0]
    return dta_out

# Func 7C: Categories for cross-tabs
int_cats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]  
int_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']

decil_cats = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 01.0, float('inf')]  
decil_labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0+']

perc_cats = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, float('inf')]  
perc_labels = ['0.00-0.01', '0.01-0.02', '0.02-0.03', '0.03-0.04', '0.04-0.05', '0.05-0.06', '0.06-0.07', '0.07-0.08', '0.08-0.09', '0.09-0.10', '0.10+']

neg_perc_cats = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, float('inf')]  
neg_perc_labels = ['-0.03-0.02', '-0.02-0.01', '-0.01-0.00', '0.00-0.01', '0.01-0.02', '0.02-0.03', '0.03-0.04', '0.04-0.05', '0.05+']

# Func 7D: Decile summary
def decile_summary(dta_in, var: str):
    """ Produces quick summary with deciles
        usage: decile_summary(stat_optmz_sample, 'dist_entry_to_lod')
    """
    summary = dta_in[var].describe()
    deciles = dta_in[var].quantile([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    combined_summary = pd.concat([summary, deciles])
    return combined_summary

# Func 8A: Simple line chart
def line_chart(dta_in, date_var, line1, line2, lab1, lab2):
    """ Plots only two lines for now
        sample usage: line_chart(pnl_byDate_0006, 'normed_date', 'comp_ret_g', 'comp_ret_n', '% ret', 'G & N compounded ret')
    """    
    plt.plot(dta_in[date_var], dta_in[line1], label = line1)
    plt.plot(dta_in[date_var], dta_in[line2], label = line2)
    plt.xlabel('Date')
    plt.ylabel(lab1)
    plt.title(lab2)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation = 45)
    plt.show()
    return

# Func 8B: Simple line chart grid
def line_chart_grid(ax, dta_in, date_var, line1, line2, lab1, lab2):
    """ Plots only two lines for now
        sample usage: line_chart(pnl_byDate_0006, 'normed_date', 'comp_ret_g', 'comp_ret_n', '% ret', 'G & N compounded ret')
    """    
    ax.plot(dta_in[date_var], dta_in[line1], label = line1)
    ax.plot(dta_in[date_var], dta_in[line2], label = line2)
    ax.set_xlabel('Date')
    ax.set_ylabel(lab1)
    ax.set_title(lab2)
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    return

# Func 9: Plots: dot, box or kernel
def grid_plot(dta_in, cols: list, size = 7, plot_type = 'box'):
    """ 3 different kinds of plots: box, dot, and kernel
        sample usage: grid_plot(static_rvw, ['entry_price', 'ATR', 'RVOL'], 7, 'kernel')
    """    

    plot_type = plot_type.lower()
    valid_choices = ['dot', 'kernel', 'box']
    if plot_type not in valid_choices:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be one of {', '.join(valid_choices)}")

    keep_cols = cols
    df = dta_in[keep_cols]
    num_vars = df.shape[1]
    # Calculate grid size
    #grid_size = int(np.ceil(np.sqrt(num_vars)))
    grid_size = int(np.ceil(np.sqrt(num_vars)))
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(size, size))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each variable
    for i, var in enumerate(df.columns):
        if plot_type == 'dot':
            sns.stripplot(data=df, x=var, ax=axes[i], size=4, jitter=True, color='blue', alpha=0.5)
        elif plot_type == 'kernel':    
            sns.kdeplot(data=df, x=var, ax=axes[i], color='purple', linewidth=2.5)
        else:
            sns.boxplot(data=df, y=var, ax=axes[i])
        axes[i].set_title(var)
        axes[i].grid(True) 

    # Remove any empty subplots
    #for j in range(i + 1, grid_size * grid_size):
    for j in range(num_vars, grid_size * grid_size):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Func 10: Scatter plots against calendar variables
def scatter_plot(dta_in, x_col: str, y_cols: list = ['year', 'week_day', 'month', 'entry_hr', 'quarter_hr'], size: int = 15):

    num_vars = len(y_cols)
    #grid_size = int(np.ceil(np.sqrt(num_vars)))
    grid_size = 5
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(size, size))
    axes = axes.flatten()

    # Plot each Y variable against the same X variable
    for i, y_col in enumerate(y_cols):
        sns.scatterplot(data=dta_in, x=x_col, y=y_col, ax=axes[i], color='green')
        axes[i].grid(True)

    # Remove any empty subplots
    for j in range(num_vars, grid_size * grid_size):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Func 11: Creates distance variables
def create_distance(pref: str , dta_in: pd.DataFrame, var_from: str, var_to: str, var_norm: str) -> pd.DataFrame:
    """ Creates a new column based on the operation and returns the modified dataframe
        pref: prefix for variable names
        dta_in: name of data in
        var_from: reference variable (distance from)
        var_to: reference variable (distance to)
        var_norm: divisor (variable to normalize distance)
    """
    dta_in[f'{pref}_{var_from}_{var_to}'] = (dta_in[var_from] - dta_in[var_to]) / dta_in[var_norm]
    return dta_in

# Func 12: Dataframe Summary
_summary_store = {
    "df": pd.DataFrame(columns=["Name", "obs", "ncols"])
}

# Func 12A:
def append_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Appends a summary row with the DataFrame name, number of observations,
    and number of columns to the internal summary store.

    Parameters:
        df (pd.DataFrame): The dataframe to summarize.
        name (str): The name to label the dataframe.

    Returns:
        pd.DataFrame: The updated summary dataframe.
    """
    _summary_store["df"] = pd.concat([
        _summary_store["df"],
        pd.DataFrame({
            "Name": [name],
            "obs": [len(df)],
            "ncols": [df.shape[1]]
        })
    ], ignore_index=True)

    return _summary_store["df"]

# Func 12B:
def get_summary() -> pd.DataFrame:
    """
    Returns the current summary dataframe.

    Returns:
        pd.DataFrame: The summary dataframe.
    """
    return _summary_store["df"]

# Func 12C:
def reset_summary() -> None:
    """
    Resets the summary dataframe to empty.
    """
    _summary_store["df"] = pd.DataFrame(columns=["Name", "obs", "ncols"])

# Func 13:
def count_outliers_by_std(df: pd.DataFrame, columns: list, thresholds: list = [2, 3, 4, 5, 6, 7]) -> pd.DataFrame:
    result = {}

    for col in columns:
        if col not in df.columns:
            continue  # skip missing columns

        series = df[col].dropna()
        mean = series.mean()
        std = series.std()

        result[col] = {
            f"{n}sd": ((series - mean).abs() > n * std).sum()
            for n in thresholds
        }

    return pd.DataFrame.from_dict(result, orient="index")