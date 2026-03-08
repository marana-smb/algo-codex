# as of 10/4/2024

import numpy as np
import pandas as pd
from typing import List, Dict
import math
from datetime import datetime, date
from IPython.display import display, HTML
import pyfolio as pf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

## 0.2 DASHBOARD FUNCTIONS
### Two functions: dashboard() and short_dashboard()
### dashboard(): aggregates trade data into bysymbol/date and bysymbol data for analysis; creates dashboard
### short_dashboard(): creates dashboard from bysymbol/date data ALREADY created (uses create_bydate() function)

# Func 1: Creating data by Symbol and Date (for feature engineering)
def create_bysymboldate_data(dta_in: pd.DataFrame, by_cols: List, agg_dict: Dict) -> pd.DataFrame:

    dta_out = dta_in.groupby(by_cols).agg(agg_dict).reset_index()
    dta_out['wins'] = (dta_out['pl_n'] > 0).astype(int)
    dta_out['share_pl_n'] = dta_out['pl_n'] / abs(dta_out['matched_shares'])
    dta_out['week_day'] = pd.to_datetime(dta_out['normed_date']).dt.dayofweek
    dta_out['year'] = pd.to_datetime(dta_out['normed_date']).dt.year 
    dta_out['month'] = pd.to_datetime(dta_out['normed_date']).dt.month 
    dta_out['year_month'] = dta_out['entry_time'].dt.strftime('%Y-%m')
    dta_out['am_pm'] = dta_out['entry_time'].dt.strftime('%p')
    dta_out['entry_hr'] = dta_out['entry_time'].dt.hour

    return dta_out

# Func 2: Creating data by Date (for time series analysis)
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

### Func 3: Producing Strats dashboard, & feature-engineered tables
def dashboard(dta_in: pd.DataFrame, comm: float, bpower: float, mod_id: str, long_short: int, dash_type: str = 'long'):
    """ comm: commission (in dollars)
        bpower: buying power 
        mod_id: an arbitraty id that will be used as colum label ('00001')
        long_short: 1 / -1 depending on the setup
        dash_type: 'long' (default) calculates "by-symbol date"; short, calculates "by-date"
        sample usage: stratxxx, pnl_symboldate_xxx, pnl_date_xxx = dashboard_xxx(event_data, 3.5, 1000000, 'no_optmz', 1)
    """  
    global mod_spec    

    if dash_type == 'long':
        event_data_add = dta_in.copy()
        event_data_add['pl_g'] = ((event_data_add.exit_price - event_data_add.entry_price) * event_data_add.exit_shares * event_data_add.exit_side * long_short)
        
        if round(sum(event_data_add['pl_g']) - sum(event_data_add['mtm_pl']),6) < 0.01:
            print("NOTE: Gross P/L calc matches original")
        else:
            print("NOTE: Gross P/L calc error!")
        
        # Commision in $ per 1K shares / 1 trip as we are summing per row (each matched_shares contains a buy or sell trip)
        # BP, buying power for return calculation
        comm = comm
        trips = 1
        bp = bpower
        event_data_add['fees'] = abs(event_data_add.matched_shares) * (comm / 1000) * trips
        event_data_add['pl_n'] = event_data_add.pl_g - event_data_add.fees * long_short
        
        by_cols = ['normed_date', 'symbol']
        # Removes top two column name because those are by_cols
        datacol_names = list(event_data_add.columns)[2:]
        # sum_cols = ['mtm_pl', 'entry_pl', 'matched_shares', 'entry_fees', 'exit_fees', 'exit_shares', 'pl_g', 'pl_n', 'fees']
        sum_cols = ['mtm_pl', 'entry_pl', 'matched_shares', 'entry_side', 'entry_fees', 'exit_fees', 'exit_shares', 'pl_g', 'pl_n', 'fees']
        # Removes last column (model spec is a string)
        max_cols = [item for item in datacol_names if item not in sum_cols][:-1]
        # Creating aggregation dictionary (sum or max)
        sum_dict = {key: 'sum' for key in (sum_cols)}
        max_dict = {key: 'max' for key in (max_cols)}
        agg_dict = {**sum_dict, **max_dict}

        # Data Aggregation
        pnl_bySymbolDate = create_bysymboldate_data(event_data_add, by_cols, agg_dict)
        pnl_byDate = create_bydate_data(pnl_bySymbolDate, sum_dict, bp)
        ##
        # Calculating metrics for Dashboard
        #Basic return stats
        ret_arr = np.array(pnl_byDate['ret_n'])
        per_stats = pf.timeseries.perf_stats(ret_arr)
        
        # Data Dictionary for metrics (second part)
        metrics_dict = {
            'model id': mod_id,  # Direct value
            'start date': pnl_byDate.iloc[0]['normed_date'],  # Direct value
            'end date': pnl_byDate.iloc[-1]['normed_date'],  # Direct value
            'total days': (pnl_byDate.iloc[-1]['normed_date'] - pnl_byDate.iloc[0]['normed_date']).days,  # Direct value
            'trade days': len(pnl_byDate),  # Direct value
            'total trades': event_data_add.shape[0], # Direct value
            'trades per day':event_data_add.shape[0] / len(pnl_byDate), # Direct value
            'max draw dollar': pnl_bySymbolDate.loc[pnl_bySymbolDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max draw date': pnl_bySymbolDate.loc[pnl_bySymbolDate['pl_n'].idxmin()]['normed_date'],  # Direct value    
            'max drawday dollar': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max drawday date': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['normed_date'],  # Direct value
            'share pl': sum(pnl_bySymbolDate['pl_n']) / sum(pnl_bySymbolDate['matched_shares'] * 1),  # Direct value
            'win rate': sum(pnl_bySymbolDate['wins']) / len(pnl_bySymbolDate),  # Direct value
            'ret_n': pnl_byDate.iloc[-1]['comp_ret_n'] - 1,  # Direct value
            'ret_g': pnl_byDate.iloc[-1]['comp_ret_g'] - 1,  # Direct value
            'annual ret': round(per_stats[0], 3),  # Direct value
            'sharpe': round(per_stats[3], 3),  # Direct value
            'calmar': round(per_stats[4], 3),  # Direct value
            'sortino': round(per_stats[8], 3),  # Direct value
            'skewness': pnl_byDate['ret_n'].skew(),  # Direct value
            'kurtos': pnl_byDate['ret_n'].kurtosis(),  # Direct value
            'vol': pnl_byDate['ret_n'].std(),  # Direct value
            'annualized vol': pnl_byDate['ret_n'].std() * math.sqrt(252),  # Direct value
            'daily var': round(per_stats[12], 3),  # Direct value
            'max draw': round(per_stats[6], 3),  # Direct value
            #'mod_spec': dta_in.iloc[-1]['ModelSpec']  # Direct value
            }

        # Convert the dictionary into a two-column DataFrame
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['metric', 'value'])

        # Now concatenate the pnl_tot DataFrame (we will rename columns to fit the format)
        pnl_tot_df = pnl_bySymbolDate.agg(sum_dict).reset_index()
        pnl_tot_df.rename(columns={'index': 'metric', 0: 'value'}, inplace=True)

        # Concatenate pnl_tot_df and metrics_df
        dta_out = pd.concat([metrics_df, pnl_tot_df], axis=0, ignore_index=True)

        return dta_out, pnl_bySymbolDate, pnl_byDate
    
    else:
        comm = comm
        trips = 1
        bp = bpower
        by_cols = ['normed_date', 'symbol']
        # Removes top two column name because those are by_cols
        datacol_names = list(dta_in.columns)[2:]
        sum_cols = ['mtm_pl', 'entry_pl', 'matched_shares', 'entry_fees', 'exit_fees', 'exit_shares', 'pl_g', 'pl_n', 'fees']
        # Removes last column (model spec is a string)
        max_cols = [item for item in datacol_names if item not in sum_cols][:-1]
        # Creating aggregation dictionary (sum or max)
        sum_dict = {key: 'sum' for key in (sum_cols)}
        max_dict = {key: 'max' for key in (max_cols)}
        agg_dict = {**sum_dict, **max_dict}

        # Data Aggregation
        pnl_byDate = create_bydate_data(dta_in, sum_dict, bp)
        ##
        # Calculating metrics for Dashboard
        #Basic return stats
        ret_arr = np.array(pnl_byDate['ret_n'])
        per_stats = pf.timeseries.perf_stats(ret_arr)
        
        # Data Dictionary for metrics (second part)
        metrics_dict = {
            'model id': mod_id,  # Direct value
            'start date': pnl_byDate.iloc[0]['normed_date'],  # Direct value
            'end date': pnl_byDate.iloc[-1]['normed_date'],  # Direct value
            'total days': (pnl_byDate.iloc[-1]['normed_date'] - pnl_byDate.iloc[0]['normed_date']).days,  # Direct value
            'trade days': len(pnl_byDate),  # Direct value
            'max drawday dollar': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max drawday date': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['normed_date'],  # Direct value
            'ret_n': pnl_byDate.iloc[-1]['comp_ret_n'] - 1,  # Direct value
            'ret_g': pnl_byDate.iloc[-1]['comp_ret_g'] - 1,  # Direct value
            'annual ret': round(per_stats[0], 3),  # Direct value
            'sharpe': round(per_stats[3], 3),  # Direct value
            'calmar': round(per_stats[4], 3),  # Direct value
            'sortino': round(per_stats[8], 3),  # Direct value
            'skewness': pnl_byDate['ret_n'].skew(),  # Direct value
            'kurtos': pnl_byDate['ret_n'].kurtosis(),  # Direct value
            'vol': pnl_byDate['ret_n'].std(),  # Direct value
            'annualized vol': pnl_byDate['ret_n'].std() * math.sqrt(252),  # Direct value
            'daily var': round(per_stats[12], 3),  # Direct value
            'max draw': round(per_stats[6], 3),  # Direct value
            }

        # Convert the dictionary into a two-column DataFrame
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['metric', 'value'])

        # Now concatenate the pnl_tot DataFrame (we will rename columns to fit the format)
        pnl_tot_df = dta_in.agg(sum_dict).reset_index()
        pnl_tot_df.rename(columns={'index': 'metric', 0: 'value'}, inplace=True)

        # Concatenate pnl_tot_df and metrics_df
        dta_out = pd.concat([metrics_df, pnl_tot_df], axis=0, ignore_index=True)
        #dta_out = pd.DataFrame(list(metrics_dict.items()), columns=['metric', 'value'])

        return dta_out, pnl_byDate

