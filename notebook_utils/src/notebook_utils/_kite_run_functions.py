from cloudquant.client import Client
from datetime import datetime
import pandas as pd
import time as system_time
import itertools

# Static Variables
my_token = 'eaee3716-3a15-36f8-a7f0-98477e4333a0'
# my_envir = 'https://tath.kite.cluster.ktginnovation.com/'
my_envir = 'https://kiteapi.ktginnovation.com'

##
# Function 0: Creates string formated for submission name
def sub_name(code: str) -> str:
    current_datetime = datetime.now()
    return  code + current_datetime.strftime("%m-%d %H:%M:%S")
#0__________________________________________________________________________________________________________________________________________________________________________________

# Function 1: Wrapper for client.submit() function - lays out parameters & options
def submit_backtest_ma(start_date, end_date, start_time, end_time, submission_name, strat_guid,
                 environment = my_envir,
                 symbols = ['__ALL__'], params={}, extra_symbols = ['SPY','VXX'], 
                 partial_fill = False, 
                 enable_latency = True, # Add on Options
                 python = '3.11.2.9', #  Add on Options
                 profiling = False, #  Add on Options
                 fast = False, #  Add on Options
                 bp = 10000000, #  Add on Options
                 email = False):
    
    """ non-default arguments = start_date, end_date, start_time, end_time.
        default arguments = submission_name, environment, strategy, symbols, params, extra_symbols, partial_fill, email.
        options = latencies, profiling, fast, bp and options dict{}.
        sample usage: 
            code = 'Phx_V0.3_202406 '
            submit_backtest_ma("2024-08-30", "2024-08-30", "09:30:00", "16:00:00", sub_name(code), strat_guid ='Gr8Scriptbd35b8656f844f5a96fd4a7d973090f6')
    """ 

    client = Client(master = environment, token = my_token)
    if client.strategy_exists(strat_guid) == False:
        print ("Strategy guid doesn't exit")
        return
    
    latencies = {"ack_latency":5, "fill_latency":100, "exchange_latency":5, "market_data_latency":5}  # Options

    options = {"enable_line_profiling": profiling, "multiday_simulation": False, "fast_simulation": fast, 
               "buying_power": bp, "email_on_completion": email, "python_version": python, "apply_strategy_latency":enable_latency,
               "latencies": {"ack": latencies["ack_latency"], "fill": latencies["fill_latency"], "exchange": latencies["exchange_latency"], 
                             "market_data": latencies["market_data_latency"]}
               }
    
    return client.submit(strategy = strat_guid,  start_date = start_date, end_date = end_date, 
                         symbols = symbols, start_time = start_time, end_time = end_time,
                         strategy_params = params, extra_symbols = extra_symbols, positions = None,
                         name = submission_name,  description = submission_name, partial_fill = partial_fill,
                         options = options, email = email)
#1__________________________________________________________________________________________________________________________________________________________________________________

#Function 2: Wrapper for client.submission_delete() to delete backtest result
def delete_submissions_ma(hash_to_delete: str, environment = my_envir):
    
    client = Client(master = environment, token = my_token)
        
    print ('deleting submissions...')
    client.submission_delete(hash_to_delete)
    print ('submissions deleted')
#2__________________________________________________________________________________________________________________________________________________________________________________

#Function 3: Wrapper for client.get_trades_df function
def get_backtest_trades_ma(submission_guid: str, select_days: str = None, exclude_days: str = None, save: bool = False, path: str = None, name: str = None, 
                        environment = my_envir,
                        show_df: bool = True):
    
    client = Client(master = environment, token = my_token)
    
    if isinstance(exclude_days, list):
        remove_days = list(exclude_days) 
        print (f'Number of remove days inputted: {len(remove_days)}')
    
    df = []
    
    try:
        df = client.get_trades_df([submission_guid])
    
    except:
        print ('no such Submission GUID')
        return
                        
    obs = len(df)
    
    if obs==0:
        print ('No trade data available')
        
    if obs>=1:
        
        print (f'Created DF with {obs} rows')
        
        df.drop(columns='account', inplace=True)
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        
        start = df["date"].min().strftime("%m%d%y")
        end = df["date"].max().strftime("%m%d%y")
        
        submission_name = f'{name}_{start}_{end}'
        df["Submission_name"] = submission_name
        
        # Choose days:
        if isinstance(select_days, list):
            
            select_days_df = pd.DataFrame(select_days)
            print (f'len of select_days_df: {len(select_days_df)}')
            
            select_days_df.rename(columns={0:"date"}, inplace=True)
            select_days_df["date"] = pd.to_datetime(select_days_df["date"]).dt.date
            
            select_days_list = list(select_days_df["date"])
            print (f'Number of days to include in backtest: {len(select_days_list)}')
            #print (select_days_list)
          
            df = df[df["date"].isin(select_days)]
        
        # Exclude days:
        if isinstance(exclude_days, list):
            remove_days = list(exclude_days) 
            remove_days_df = pd.DataFrame(remove_days)
            #print (f'len of remove_days_df: {len(remove_days_df)}')
            
            remove_days_df.rename(columns={0:"date"}, inplace=True)
            remove_days_df["date"] = pd.to_datetime(remove_days_df["date"]).dt.date
            
            remove_days_list = list(remove_days_df["date"])
            print (f'Number of days to exclude from backtest: {len(remove_days_list)}')
            #print (remove_days_list)
          
            df = df[~df["date"].isin(remove_days)]
            
                 
        if save:
            print (f'Saving csv with {obs} rows...')
            df.to_csv(f'{path}{submission_name}.csv')
            print ('Save complete')
    
    if show_df:
        return df
#3__________________________________________________________________________________________________________________________________________________________________________________

#Function 4: Wrapper for client.status(submission=guid) function
def check_backtest_status_ma(guid:str, environment = my_envir, check_interval: int = 1, select_days: str = None, exclude_days: str = None, save_trades: bool = False,
                          path: str = None, 
                          name: str = None,
                          show_df: bool = True,
                          ignore_failed: bool = True):
    
    client = Client(master=environment, token=my_token)
    try:
        submission_details = client.status(submission=guid)[guid]
        jobs = len(submission_details)
        print (f'{jobs} jobs submitted')
        backtest_finished = 0
        
    except:
        print ("No such submission GUID")
        return
    
    backtest_successful = 0
    df = []
    while backtest_finished == 0:
        
        submission_details = client.status(submission=guid)[guid]
        success_count = 0
        pending_count = 0
        failed_count = 0
        
        print ('Checking backtest status...')
        for i in range(0, jobs):
            submission_status = submission_details[i]['status']

            if submission_status == 'success':
                success_count += 1

            if submission_status == 'failure':
                failed_count += 1
                print ('check failed jobs')
                
            if submission_status in ('waiting_room', 'running', 'holding_pen'):
                pending_count += 1
                
        print (f'{success_count} Successful; {failed_count} Failed; {pending_count} Pending')
                                     
        if pending_count == 0:
            print (f'{success_count} jobs successfully completed')
            print (f'{failed_count} jobs failed')
            backtest_finished = 1
    
        if backtest_finished == 0:
            system_time.sleep(check_interval*60)
        
    if backtest_finished == 1:
        if failed_count == 0:
            backtest_successful = 1
            print ('Creating DF for trades')
        
        if failed_count == 0 or ignore_failed:
            df = get_backtest_trades_ma(submission_guid=guid,
                                     environment=environment,
                                     select_days=select_days,
                                     exclude_days=exclude_days,
                                     save=save_trades,
                                     path=path,
                                     name=name,
                                     show_df=show_df)
        
        else:
            print (f'{failed_count} failed jobs, cannot get df')

        
    return backtest_finished, backtest_successful, df
#4__________________________________________________________________________________________________________________________________________________________________________________

#Function 5: KITE Multi-run, wrapper for submit_backtest_ma(), check_backtest_status_ma()
def submit_param_comb_ma(strat_guid: str = None,
                         environment = my_envir,
                         start_date: str = None,
                         end_date: str = None,
                         list_of_params: dict = {},
                         symbols = ['__ALL__'],
                         extra_symbols = ['SPY','VXX'],
                         python: str = 'ksim:3.12.3-2',
                         start_time: str = "09:30:00",
                         end_time: str = "16:00:00",
                         profiling: bool = False,
                         fast: bool = False,
                         bp: int = 10000000,
                         enable_latency: bool = False,
                         select_days = None,
                         exclude_days = None,
                         partial_fills: bool = False,
                         email = False,
                         name: str = None,
                         check_interval: int = 1,
                         save_df: bool = False,
                         send_all: bool = False):
    
    #----------------------------------------------------------
    cols = ['Test', 'Submission', 'Submission Date', 'Start', 'End', 'Params']
    summary_df = pd.DataFrame(columns=cols)

    param_combs = len(list_of_params)
    
    if len(list_of_params)==0:
        print ('No parameters to iterate')
        return
    
    keys = list_of_params.keys()
    values = list_of_params.values()
    combinations = list(itertools.product(*values))
    param_combinations = [{key: value for key, value in zip(keys, combo)} for combo in combinations]
    n_combinations = len(param_combinations)
    
    count = 0
    #---------------------------------------------------------
    if send_all:
        
        for param_dict in param_combinations:
            count += 1
            print ("--------------------------------------------")
            print (f'Submitting backtest {count} or {n_combinations} for params: {param_dict}')

            #add on
            param_str = "_".join(str(value) for value in param_dict.values())
            submission_name = f'{name}_{param_str}_{count}'

            submission_guid = submit_backtest_ma(start_date = start_date, end_date = end_date, start_time = start_time, end_time = end_time, submission_name = submission_name, 
                                                strat_guid = strat_guid, environment = environment, symbols = symbols, params = param_dict, extra_symbols = extra_symbols, 
                                                partial_fill = partial_fills, enable_latency = enable_latency, # Add on Options
                                                python = python, #  Add on Options
                                                profiling = profiling, #  Add on Options
                                                fast = fast, #  Add on Options
                                                bp = bp, #  Add on Options
                                                email = email)

        system_time.sleep(15)
    
    #----------------------------------------------------------
    
    if not send_all:
        
        for param_dict in param_combinations:
            count += 1
            print ("--------------------------------------------")
            print (f'Submitting backtest {count} or {n_combinations} for params: {param_dict}')

            #add on
            param_str = "_".join(str(value) for value in param_dict.values())
            submission_name = f'{name}_{param_str}_{count}'

            submission_guid = submit_backtest_ma(start_date = start_date, end_date = end_date, start_time = start_time, end_time = end_time, submission_name = submission_name, 
                                                strat_guid = strat_guid, environment = environment, symbols = symbols, params = param_dict, extra_symbols = extra_symbols, 
                                                partial_fill = partial_fills, enable_latency = enable_latency, # Add on Options
                                                python = python, #  Add on Options
                                                profiling = profiling, #  Add on Options
                                                fast = fast, #  Add on Options
                                                bp = bp, #  Add on Options
                                                email = email)
            
            system_time.sleep(30)

            status = check_backtest_status_ma(guid = submission_guid, environment = environment, check_interval = check_interval, select_days = select_days,
                                           exclude_days = exclude_days, show_df = False)

            today = datetime.today().strftime('%m/%d/%Y')

            if status[0]==1:
                print ('Appending submission details to summary df')
                my_guid = str(submission_guid)

                new_row = pd.DataFrame([{'Test': count, 
                         'Submission': my_guid,
                         'Submission Date':today,
                         'Start':start_date,
                         'End':end_date,
                         'Params':param_dict}])

                summary_df = pd.concat([summary_df, new_row], ignore_index=True)

            else:
                print (f'Backtest #{count} failed, no df outputted')

            if save_df:
                try:
                    summary_df.to_csv(f'{name}_submissions_info.csv')
                except:
                    print ('Cannot save df, check if it is open')
    
    #----------------------------------------------------------
    guid_list = list(summary_df['Submission'])
    
    return guid_list, summary_df
#5__________________________________________________________________________________________________________________________________________________________________________________