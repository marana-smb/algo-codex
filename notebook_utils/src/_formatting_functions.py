from __future__ import annotations
import numpy as np
import pandas as pd
import os
from typing import Optional

#from datetime import datetime, date
#from IPython.display import display, HTML

## 0.0 UTILITY FUNCTIONS
# Func 1: negative red - style 1
def color_negative_red(val):    
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

# Func 2: style aggregator + decimals
def make_pretty(styler):
    styler.applymap(color_negative_red)
    styler.format(precision = 4, thousands = ",", decimal = ".")
    return styler

# Dict 3A: To format data
col_names = {
    'index': 'totals',
    '0': 'dollars'
    }

# Func 3B: formatting output as DataFrame
def data_format(dta, var_txt, var_name):   
    dta = pd.DataFrame({var_txt: [var_name]}).T.reset_index()
    dta.rename(columns = col_names, inplace = True)
    return dta

# Func 3A: Creating quarter hours
def quarter_hr(dt):
    minutes = dt.minute + dt.second / 60
    nearest_quarter = int(np.round(minutes / 15.0) * 15)
    new_time = dt.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(minutes=nearest_quarter)
    return new_time.strftime("%H:%M")

## 0.1 DATAFRAME CLEAN UP FUNCTIONS
# Func 4: Importing and cleaning data
def event_import(path, filename, front_cols=None, kite_cols=None, sym_ta_cols=None, extra_sym_cols=None, extra_cols=None, sec_cols=None, last_cols=None, drop_cols=None):
    """
    Imports, processes, and organizes event data from a CSV file.

    Parameters:
        path (str): Path to the CSV file directory.
        filename (str): Name of the CSV file.
        front_cols (list): List of front columns to include.
        kite_cols (list): List of kite columns to include.
        sym_ta_cols (list): List of symbol TA columns to include.
        extra_sym_cols (list): List of extra symbol columns to include.
        extra_cols (list): List of extra columns to include.
        sec_cols (list): List of secondary columns to include.
        last_cols (list, optional): List of last columns to include. Defaults to an empty list.
        drop_cols (list, optional): List of columns to drop. Defaults to an empty list.

    Returns:
        pd.DataFrame: Processed event data.
    """
    import pandas as pd  # Ensure pandas is imported

    # Set default values for optional parameters
    if front_cols is None:
        front_cols = []
    if kite_cols is None:
        kite_cols = []
    if sym_ta_cols is None:
        sym_ta_cols = []
    if extra_sym_cols is None:
        extra_sym_cols = []
    if extra_cols is None:
        extra_cols = []
    if sec_cols is None:
        sec_cols = []
    if last_cols is None:
        last_cols = []
    if drop_cols is None:
        drop_cols = []

    # Import and process the CSV file
    event_data = pd.read_csv(path + filename)
    event_data['entry_time'] = pd.to_datetime(event_data['entry_time'])
    event_data['exit_time'] = pd.to_datetime(event_data['exit_time'])
    event_data['normed_date'] = event_data['entry_time'].dt.date
    event_data.sort_index(inplace=True)

    # Reorder columns
    all_columns = ['normed_date'] + front_cols + kite_cols + sym_ta_cols + extra_sym_cols + extra_cols + sec_cols + last_cols + drop_cols
    event_data = event_data[all_columns]

    # Adjust last columns if applicable
    if last_cols:
        event_data = event_data[[col for col in event_data.columns if col != last_cols[0]] + last_cols]

    # Drop specified columns
    if drop_cols:
        event_data = event_data.drop(columns=drop_cols)

    # Rename columns
    event_data.columns = event_data.columns.str.replace('entry_collect.', '')

    # Sort the data
    event_data.sort_values(by=['entry_time', 'symbol'], ascending=[False, True], inplace=True)

    return event_data

# Func 5: Data dictionary
def generate_data_dictionary(input_filename, input_folder=None, output_filename='data_dictionary_pre.xlsx', output_folder=None):
    """
    Reads a CSV, cleans column names by removing '.' prefixes, 
    extracts data types, and exports a data dictionary to Excel.

    Parameters:
    - input_filename (str): Name of the source CSV file.
    - input_folder (str): Folder path. Defaults to user's Downloads folder if None.
    - output_filename (str): Name of the resulting Excel file.
    - output_folder (str): Folder path. Defaults to input_folder if None.
    """
    
    # 1. Set Default Paths
    # If no folder provided, default to 'Downloads'
    if input_folder is None:
        input_folder = os.path.expanduser('~/Downloads')
    
    # If no output folder provided, save in the same place as input
    if output_folder is None:
        output_folder = input_folder

    # Construct full paths safely
    input_path = os.path.join(input_folder, input_filename)
    output_path = os.path.join(output_folder, output_filename)

    try:
        # 2. Load Data
        print(f"Reading file: {input_path}...")
        df = pd.read_csv(input_path)

        # 3. Clean Names & Get Types
        # Split by '.' and take the last part
        clean_names = [col.split('.')[-1] for col in df.columns]
        
        # Get data types as strings
        data_types = df.dtypes.astype(str).values

        # 4. Construct Dictionary DataFrame
        dictionary_df = pd.DataFrame({
            'Field Name': clean_names,
            'Original Source Name': df.columns,
            'Data Type': data_types
        })

        # 5. Export
        dictionary_df.to_excel(output_path, index=False)
        print(f"Success! Dictionary saved to: {output_path}")
        
        return dictionary_df

    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}' in '{input_folder}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Func 6: Populating data dictionary with additional tags for work flow
def enrich_data_dictionary(current_dictionary_df, master_inventory_path, sheet_name='master', output_filename='data_dictionary_post.xlsx', output_folder=None):
    """
    Merges a generated data dictionary with a Master Variable Inventory Excel file.
    
    Parameters:
    - current_dictionary_df (pd.DataFrame): The dataframe created in step 1.
    - master_inventory_path (str): Full path to the Master Inventory Excel file.
    - sheet_name (str): The specific tab to read in the master file. Default is 'master'.
    - output_filename (str): Name of the final output file.
    - output_folder (str): Where to save the file. Defaults to Downloads if None.
    """
    
    # Set default output folder to Downloads if not provided
    if output_folder is None:
        output_folder = os.path.expanduser('~/Downloads')
    
    output_path = os.path.join(output_folder, output_filename)

    try:
        print(f"Loading Master Inventory from: {master_inventory_path}")
        master_df = pd.read_excel(master_inventory_path, sheet_name=sheet_name)
        
        # Perform Left Merge
        # We keep all rows from the current_dictionary_df (Left)
        merged_df = pd.merge(
            current_dictionary_df,
            master_df,
            left_on='Field Name',    # Column in your generated dictionary
            right_on='clean name',   # Column in the master inventory
            how='left'
        )

        # Export to Excel
        merged_df.to_excel(output_path, index=False)
        print(f"Success! Enriched dictionary saved to: {output_path}")
        
        # --- The Optional Check ---
        # Identify rows where 'clean name' is NaN (meaning no match found in Master)
        unmapped = merged_df[merged_df['clean name'].isnull()]
        
        if not unmapped.empty:
            count = len(unmapped)
            print(f"\n[ALERT] {count} variables from your CSV are NOT in the Master Inventory.")
            print("Examples of missing variables:")
            print(unmapped['Field Name'].head(5).to_list())
        else:
            print("\n[OK] All variables were successfully mapped to the Master Inventory.")

        return merged_df

    except FileNotFoundError:
        print(f"Error: Could not find the Master Inventory file at: {master_inventory_path}")
        return None
    except Exception as e:
        print(f"An error occurred during enrichment: {e}")
        return None

# Func 7: NaN/Inf panda data summary
def nan_inf_summary(df: pd.DataFrame, *, print_columns: bool = True, df_name: Optional[str] = None) -> pd.DataFrame:
    """
    Summarize NaN and +/-Inf issues in a DataFrame.
    ----------
    df: Input DataFrame to inspect.
    print_columns: If True, prints columns containing any NaN and any +/-Inf.
    df_name: Optional label to include in printed output.
    -----
    - Inf counts are computed only on numeric columns.
    - Output includes only columns with at least one NaN or Inf.
    """
    label = f"[{df_name}] " if df_name else ""

    # NaN counts (all columns)
    nan_count = df.isna().sum()
    nan_count = nan_count[nan_count > 0]

    # Inf/-Inf counts (numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df).sum()
    inf_count = inf_count[inf_count > 0]

    # Union of affected columns
    cols = nan_count.index.union(inf_count.index)

    summary_df = pd.DataFrame(
        {
            "NaN Count": nan_count.reindex(cols, fill_value=0),
            "Inf Count": inf_count.reindex(cols, fill_value=0),
        }
    ).astype(int)

    if print_columns:
        print(f"{label}Columns with Inf/-Inf values: {inf_count.index.tolist()}")
        print(f"{label}Columns with NaN values: {nan_count.index.tolist()}")
        print(summary_df)

    return summary_df
