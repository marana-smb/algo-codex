# as of 2/9/2025: Modified for data with on symbol (aggregating by entry_time as well)
# as of 5/10/2025: Aligning calculations to match KITE scorecard - Pyfolio not used anymore to speedup calculations1

import numpy as np
import pandas as pd
from typing import List, Dict
from IPython.display import display, HTML
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

## 0.2 DASHBOARD FUNCTIONS
### One function: dashboard() with multiple options (to calculate returns with max capital used, and to calculate a dashboard already aggregated data; complete/incomplete)

# Func 1: Creating data by Symbol and Date (for feature engineering)
def create_bysymboldate_data(dta_in: pd.DataFrame, by_cols: List, agg_dict: Dict) -> pd.DataFrame:

    dta_out = dta_in.groupby(by_cols).agg(agg_dict)

    if 'entry_time' in dta_out.columns:
        dta_out = dta_out.drop(columns=['entry_time'])
    # Reset index safely
    dta_out = dta_out.reset_index()

    dta_out['capital'] = dta_out['entry_price'] * abs(dta_out['entry_shares'])
    dta_out['max_cap'] = dta_out['capital'].max()
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
def create_bydate_data(dta_in: pd.DataFrame, sum_dict: dict, bp: float) -> pd.DataFrame:
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

    if bp == 1000000:        
        dta_out['ret_g'] = dta_out['pl_g'] / bp
        dta_out['ret_n'] = dta_out['pl_n'] / bp
    else:
        global_max_cap = dta_in['capital'].max()
        dta_out['ret_g'] = dta_out['pl_g'] / global_max_cap
        dta_out['ret_n'] = dta_out['pl_n'] / global_max_cap

    print("global_max_cap:", global_max_cap)

    dta_out['max_cap'] = global_max_cap
    dta_out['cum_ret_g'] = 1 + dta_out['ret_g']
    dta_out['cum_ret_n'] = 1 + dta_out['ret_n']
    dta_out['comp_ret_g'] = dta_out['cum_ret_g'].cumprod()
    dta_out['comp_ret_n'] = dta_out['cum_ret_n'].cumprod()
    dta_out['high_comp_ret_n'] = dta_out['cum_ret_n'].cummax()
    return dta_out

### Func 3: Producing Strats dashboard, & feature-engineered tables
def dashboard(dta_in: pd.DataFrame, comm: float, mod_id: str, long_short: int, dash_type: str = 'complete', bpower: str = 'max'):
    """ comm: commission (in dollars) 
        mod_id: an arbitraty id that will be used as colum label ('00001')
        long_short: 1 / -1 depending on the setup
        dash_type: 'complete' (default) calculates "by-symbol date"; 'by_date', calculates "by-date"
        bpower: buying power; 'max' uses capital used; else uses 1,000,000
        sample usage: strat000, pnl_symboldate_000, pnl_bydate_000 = dashboard(event_data, 3.5, 'no_optmz', -1, 'complete', 'max')
    """  
    global mod_spec    

    # Dashboard that creates by_symbol/date data as well as by_date data
    if dash_type == 'complete':
        event_data_add = dta_in.copy()
        event_data_add['pl_g'] = ((event_data_add.exit_price - event_data_add.entry_price) * event_data_add.exit_shares * event_data_add.exit_side * long_short)
        
        if round(sum(event_data_add['pl_g']) - sum(event_data_add['mtm_pl']),6) < 0.01:
            print("NOTE: Gross P/L calc matches original")
        else:
            print("NOTE: Gross P/L calc error!")
        
        # Commision in $ per 1K shares / 1 trip as we are summing per row (each matched_shares contains a buy or sell trip)
        comm = comm
        trips = 1        

        # Additional columns for sum_cols (included in dictionary...to sum by)
        event_data_add['fees'] = abs(event_data_add.matched_shares) * (comm / 1000) * trips
        event_data_add['pl_n'] = event_data_add.pl_g - event_data_add.fees * long_short

        if not bpower == 'max':
            bp = 1000000
        else:
            bp = 1
        
        by_cols = ['normed_date', 'symbol', 'entry_time']
        # Removes top two column name because those are by_cols
        datacol_names = list(event_data_add.columns)[2:]
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
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        start_date = pnl_byDate.iloc[0]['normed_date']
        end_date = pnl_byDate.iloc[-1]['normed_date']
        biz_days = len(pd.date_range(start=start_date, end=end_date, freq=us_bd))
        vol = pnl_byDate['ret_n'].std()
        pnl_byDate['ret_n_neg'] = pnl_byDate['ret_n'].where(pnl_byDate['ret_n'] < 0, 0)

        # Data Dictionary for metrics (second part)
        metrics_dict = {
            'model id': mod_id,  # Direct value
            'start date': pnl_byDate.iloc[0]['normed_date'],  # Direct value
            'end date': pnl_byDate.iloc[-1]['normed_date'],  # Direct value
            'total days': (pnl_byDate.iloc[-1]['normed_date'] - pnl_byDate.iloc[0]['normed_date']).days,  # Direct value
            'trade days': len(pnl_byDate),  # Direct value
            'total trades': event_data_add.shape[0], # Direct value
            'total trading days': biz_days, # Direct value UPDATE
            'trades per day':event_data_add.shape[0] / len(pnl_byDate), # Direct value
            'max draw dollar': pnl_bySymbolDate.loc[pnl_bySymbolDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max draw date': pnl_bySymbolDate.loc[pnl_bySymbolDate['pl_n'].idxmin()]['normed_date'],  # Direct value    
            'max drawday dollar': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max drawday date': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['normed_date'],  # Direct value
            'share pl': (sum(pnl_bySymbolDate['pl_n']) / sum(pnl_bySymbolDate['matched_shares'] * 2 * long_short))*100,  # Direct value UPDATE
            'daily win': pnl_byDate['wins'].mean(), # Direct value UPDATE
            'ret_n': pnl_byDate.iloc[-1]['comp_ret_n'] - 1,  # Direct value
            'ret_g': pnl_byDate.iloc[-1]['comp_ret_g'] - 1,  # Direct value
            'annual ret': np.log(pnl_byDate.iloc[-1]['comp_ret_n']) * (252/biz_days), # Direct value UPDATE
            'sharpe': (pnl_byDate['ret_n'].mean() / vol) * np.sqrt(252),  # UPDATE
            'sortino': (pnl_byDate['ret_n'].mean()/pnl_byDate['ret_n_neg'].std()) * np.sqrt(252) , #UPDATE
            'skewness': pnl_byDate['ret_n'].skew(),  # Direct value
            'kurtos': pnl_byDate['ret_n'].kurtosis(),  # Direct value
            'annual vol': vol * np.sqrt(252),  # Direct value
            }
        
        # Convert the dictionary into a two-column DataFrame
        metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['metric', 'value'])

        # Now concatenate the pnl_tot DataFrame (we will rename columns to fit the format)
        pnl_tot_df = pnl_bySymbolDate.agg(sum_dict).reset_index()
        pnl_tot_df.rename(columns={'index': 'metric', 0: 'value'}, inplace=True)

        # Concatenate pnl_tot_df and metrics_df
        dta_out = pd.concat([metrics_df, pnl_tot_df], axis=0, ignore_index=True)

        return dta_out, pnl_bySymbolDate, pnl_byDate
    
    # Dashboard that skips by_symbol/date data creation because it has been created already (most work is done on data that has been aggregate by symbol/date)
    ## By symbol/date works because we (1) trade one symbol multiple times x day; (2) multiple symbols one time per day; as opposed to trading multiple symbols, multiple times per day
    else:
        comm = comm
        trips = 1

        if not bpower == 'max':
            bp = 1000000
        else:
            bp = 1

        by_cols = ['normed_date', 'symbol']
        # Removes top two column name because those are by_cols
        datacol_names = list(dta_in.columns)[2:]
        sum_cols = ['mtm_pl', 'entry_pl', 'matched_shares', 'entry_side', 'entry_fees', 'exit_fees', 'exit_shares', 'pl_g', 'pl_n', 'fees']
        
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
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        start_date = pnl_byDate.iloc[0]['normed_date']
        end_date = pnl_byDate.iloc[-1]['normed_date']
        biz_days = len(pd.date_range(start=start_date, end=end_date, freq=us_bd))
        vol = pnl_byDate['ret_n'].std()
        pnl_byDate['ret_n_neg'] = pnl_byDate['ret_n'].where(pnl_byDate['ret_n'] < 0, 0)

        # Data Dictionary for metrics (second part)
        metrics_dict = {
            'model id': mod_id,  # Direct value
            'start date': pnl_byDate.iloc[0]['normed_date'],  # Direct value
            'end date': pnl_byDate.iloc[-1]['normed_date'],  # Direct value
            'total days': (pnl_byDate.iloc[-1]['normed_date'] - pnl_byDate.iloc[0]['normed_date']).days,  # Direct value
            'trade days': len(pnl_byDate),  # Direct value
            'total trading days': biz_days, # Direct value UPDATE
            'max draw dollar': dta_in.loc[dta_in['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max draw date': dta_in.loc[dta_in['pl_n'].idxmin()]['normed_date'],  # Direct value    
            'max drawday dollar': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['pl_n'],  # Direct value
            'max drawday date': pnl_byDate.iloc[pnl_byDate['pl_n'].idxmin()]['normed_date'],  # Direct value
            'share pl': (sum(dta_in['pl_n']) / sum(dta_in['matched_shares'] * 2 * long_short)) * 100,  # Direct value UPDATE
            'daily win': pnl_byDate['wins'].mean(), # Direct value UPDATE
            'ret_n': pnl_byDate.iloc[-1]['comp_ret_n'] - 1,  # Direct value
            'ret_g': pnl_byDate.iloc[-1]['comp_ret_g'] - 1,  # Direct value
            'annual ret': np.log(pnl_byDate.iloc[-1]['comp_ret_n']) * (252/biz_days), # Direct value UPDATE
            'sharpe': (pnl_byDate['ret_n'].mean() / vol) * np.sqrt(252),  # UPDATE
            'sortino': (pnl_byDate['ret_n'].mean()/pnl_byDate['ret_n_neg'].std()) * np.sqrt(252) , #UPDATE
            'skewness': pnl_byDate['ret_n'].skew(),  # Direct value
            'kurtos': pnl_byDate['ret_n'].kurtosis(),  # Direct value
            'annual vol': vol * np.sqrt(252),  # Direct value
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

