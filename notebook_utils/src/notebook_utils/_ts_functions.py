# Standard libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings('ignore')

###
## Simplified MCPT
#

def mcp_test(df, column1, column2, n_permutations = 10000, plot = False):
    """
    Perform a Monte Carlo Permutation Test to compare the means of two columns in a single DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column1: The name of the first column to be used for comparison.
    - column2: The name of the second column to be used for comparison.
    - n_permutations: The number of permutations to perform (default is 10000).
    - plot: If True, plots the distribution of permuted differences (default is False).

    Returns:
    - observed_diff: The observed difference in means.
    - p_value: The p-value indicating the significance of the observed difference.

    Usage:
    - observed_diff, p_value = mcp_test(df01, 'ret_g', 'NASDAQCOM_chg', plot = True) 
    - observed_diff, p_value = mcp_test(df01, 'ret_g', 'SP500_chg') 
    """
    # Step 1: Extract the relevant columns
    data1 = df[column1].dropna()
    data2 = df[column2].dropna()
    
    # Ensure both columns have the same length or handle alignment
    if len(data1) != len(data2):
        print("Warning: Columns have different lengths, aligning on index.")
        combined_df = pd.concat([data1, data2], axis=1).dropna()
        data1 = combined_df[column1]
        data2 = combined_df[column2]

    # Step 2: Calculate the observed difference in means
    observed_diff = data1.mean() - data2.mean()
    
    # Step 3: Combine both data sets
    combined_data = pd.concat([data1, data2])
    
    # Step 4: Initialize a list to store permuted differences
    permuted_diffs = np.zeros(n_permutations)
    
    # Step 5: Perform permutations
    for i in range(n_permutations):
        # Shuffle the combined data
        shuffled_data = combined_data[1:].sample(frac=1).reset_index(drop=True)
        
        # Prepend the first return to the shuffled data
        fixed_data = pd.concat([combined_data.iloc[[0]], shuffled_data]).reset_index(drop=True)
        
        # Split into two groups
        perm_data1 = fixed_data[:len(data1)]
        perm_data2 = fixed_data[len(data1):]
        
        # Calculate the difference in means for the permuted data
        perm_diff = perm_data1.mean() - perm_data2.mean()
        permuted_diffs[i] = perm_diff
    
    # Step 6: Calculate the p-value
    p_value = np.sum(permuted_diffs >= observed_diff) / n_permutations
    
    # Optional: Plot the distribution of permuted differences
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(permuted_diffs, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Permuted Differences')
        plt.xlabel('Difference in Means')
        plt.ylabel('Frequency')
        plt.legend(['Observed Difference', 'Permuted Differences'])
        plt.show()
    
    return observed_diff, p_value

###
## Implements ARIMA test
#

def arima_analysis(df: pd.DataFrame, column: str, arima_order: tuple = (1,0,1)):
    """
    Fits an ARIMA model to a specified time series column in a DataFrame and returns a summary of the model.
    df : pandas.DataFrame
    column : The name of the column in the DataFrame that contains return vector.
    arima_order : The default is ARIMA(1, 0, 1).
    usage: arima_analysis(df01,'ret_g')
    """
    model = sm.tsa.ARIMA(df[column], order=arima_order).fit(cov_type='robust')
    return model.summary()

###
## Plots correlogram
#

def plot_correlogram(df, column, lags=40):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of a time series column.
    df : The DataFrame containing the time series data.
    column : The name of the column in the DataFrame that contains the time series data to plot.
    lags : Optional. The number of lags to include in the plot. The default is 40.
    usage: plot_correlogram(df01, 'ret_g', lags = 20)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot ACF
    plt.subplot(121)
    plot_acf(df[column], lags=lags, ax=plt.gca())
    plt.title('Autocorrelation Function')
    plt.xticks(np.arange(0, lags+1, 1))  # Set x-axis ticks at every 1 unit
    
    # Plot PACF
    plt.subplot(122)
    plot_pacf(df[column], lags=lags, ax=plt.gca())
    plt.title('Partial Autocorrelation Function')
    plt.xticks(np.arange(0, lags+1, 1))  # Set x-axis ticks at every 1 unit
    
    plt.tight_layout()
    plt.show()

###
## Performs LB test
#

def ljung_box_test(df, column, lags=10):
    """
    df : The DataFrame containing the time series data.
    column : The name of the column in the DataFrame that contains the time series data to test.
    lags : The number of lags to test for autocorrelation. The default is 10.
    usage: ljung_box_results = ljung_box_test(df01, 'ret_g', lags=20).
    """
    lb_test = acorr_ljungbox(df[column], lags=lags, return_df=True)
    return lb_test
