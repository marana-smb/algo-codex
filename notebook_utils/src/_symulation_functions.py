import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, entropy
import random
import matplotlib.pyplot as plt
import textwrap

# Func 23A: Calculates optimal f* from return vector
def static_optimal_f(dta_in: pd.DataFrame, bounds: tuple, method: str ='L-BFGS-B'):
    """
    Calculates optimal f* from return vector
    usage: opt_f, _, gat, _ = static_optimal_f(pnl_date_0005_1a['ret_n'], (0,1)).
    """
    trades = np.array(dta_in)
    dd = np.min(trades)

    def objective(f):
        twr = np.prod([1 + f * (-trade / dd) for trade in trades])
        return -twr  # Negate to maximize

    initial_guess = 0.1
    result = minimize(objective, initial_guess, bounds=[bounds], method ='L-BFGS-B')

    if not result.success:
        print("Optimization failed:", result.message)  # Print the error message
        return None, None, None, None  # Return None values

    opt_f = round(result.x[0], 4)
    max_twr = round(-result.fun, 4)
    cagr = round((max_twr ** (1 / np.count_nonzero(trades)) - 1), 4)
    gat = round(cagr * -dd / opt_f, 4)
    
    return opt_f, max_twr, gat, cagr

# Func 23B: Calculates bootstraped optimal f* from return vector
def bootstrap_optimal_f(returns, num_boot=1000, seed=123):
    """
    Bootstraps a return series to calculate Optimal f
    returns (np.array): Array of historical returns.
    num_boot (int): Number of bootstrap samples.
    usage: avg_f = bootstrap_optimal_f(pnl_date_0005_1a['ret_n']).
    """
    if seed is not None:
        np.random.seed(seed)
    
    f_bootstrap = []
    for _ in range(num_boot):
        # Resample with replacement from returns
        sample = np.random.choice(returns, size=len(returns), replace=True)

        # Calculate VaR and CVaR for this sample
        trades = np.array(sample)
        dd = np.min(trades)

        def objective(f):
            twr = np.prod([1 + f * (-trade / dd) for trade in trades])
            return -twr  # Negate to maximize

        # Set initial guess and bounds for f
        initial_guess = 0.1
        bounds = [(0, 1)]  # Adjust as needed for exploration of f's range
        result = minimize(objective, initial_guess, bounds=bounds)
        optimal_f = result.x[0]
        f_bootstrap.append(optimal_f)

    # Compute average across all bootstrapped samples
    avg_f = round(np.mean(f_bootstrap),4)

    return avg_f

# Func 24: Creates monte carlo runs from empirical returns
def monte_carlo_empirical(dta_in: pd.DataFrame, scale: int=10, num_simulations: int=10000, time_horizon: int=252, initial_investment: int=1, chart: bool=True):
    """
    Monte Carlo simulation using empirical distribution (bootstrapping historical returns).
        returns (pd.Series): Historical returns series.
        num_simulations (int): Number of simulations to run.
        time_horizon (int): Number of periods (e.g., 252 trading days for one year).
        initial_investment (float): Starting amount of the investment.
        Returns: Final simulated portfolio values after the time horizon.
        Usage: final_values = monte_carlo_empirical(pnl_date_0005_1a['ret_n']).
    """
    # Initialize an array to store the final portfolio values
    final_values = np.zeros(num_simulations)
    returns = dta_in * scale
    
    for i in range(num_simulations):
        # Randomly sample from the historical returns with replacement
        simulated_returns = np.random.choice(returns, size=time_horizon, replace=True)
        # Calculate the cumulative return path
        cumulative_return = np.cumprod(1 + simulated_returns)  # Cumulative product for growth
        # Compute final portfolio value
        final_values[i] = initial_investment * cumulative_return[-1]

    if chart:
        mean_val = np.mean(final_values)
        std_dev = np.std(final_values)
        x_values = np.linspace(min(final_values), max(final_values), 1000)
        y_values = norm.pdf(x_values, mean_val, std_dev) * len(final_values) * (max(final_values) - min(final_values)) / 75

        # Plot the histogram of the final portfolio values
        plt.figure(figsize=(10, 6))
        plt.hist(final_values, bins=75, color='skyblue', edgecolor='blue', alpha=0.7)

        # Overlay the mean and percentiles as vertical lines
        plt.axvline(mean_val, color='green', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
        plt.axvline(np.percentile(final_values, 5), color='orange', linestyle='dashed', linewidth=1, label=f'5th Percentile (VaR): {np.percentile(final_values, 5):.2f}')
        plt.axvline(np.percentile(final_values, 95), color='orange', linestyle='dashed', linewidth=1, label=f'95th Percentile: {np.percentile(final_values, 95):.2f}')

        # Plot the normal distribution overlay as a dotted line
        plt.plot(x_values, y_values, color='black', linestyle='dotted', linewidth=1, label='Normal Distribution Fit')

        # Labeling
        plt.title('Distribution of Final Portfolio Values - Empirical Monte Carlo Simulation')
        plt.xlabel('Final Portfolio Value')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.grid(False, linestyle='--', alpha=0.6)
        plt.show()
    
    return final_values

# Func 25: Monte Carlo permutation test
def price_permute(dta_in: pd.DataFrame, index: int, scale: int = 1, seed: int = None):
    """Usage: perpx = price_permute(pnl_date_0005_1a['ret_n'], 1, 10)
    """
    n_prices = len(dta_in)
    permute_index = index + 1
    
    # Prepare scaled prices list and compute initial changes
    prices = (dta_in * scale).tolist()
    basis_price = prices[index]
    changes = np.zeros(n_prices)

    # Calculate changes from prior prices
    for iprice in range(permute_index, n_prices):
        changes[iprice] = prices[iprice] - prices[iprice - 1]
    
    if seed is not None:
        random.seed(seed)

    # Shuffle changes starting from permute_index
    changes_to_shuffle = changes[permute_index:]
    random.shuffle(changes_to_shuffle)
    changes[permute_index:] = changes_to_shuffle

    # Rebuild prices based on permuted changes
    prices[permute_index - 1] = basis_price
    for iprice in range(permute_index, n_prices):
        prices[iprice] = prices[iprice - 1] + changes[iprice]
    prices_df = pd.DataFrame(prices, columns=['perm_px'])

    return prices_df

# Func 26: Monte Carlo permutation test
def mc_pt(dta_in: pd.DataFrame, scale: int=1,num_per=1000, metric='mean'):
    """
    Perform a permutation test on strategy returns.
        strategy_returns (pd.Series): Series of returns from the strategy.
        num_permutations (int): Number of permutations to run.
        metric (str): Performance metric to test; options are 'mean' or 'sharpe'.
        Returns: P-value indicating the significance of the observed metric.
        Usage: p_value = mc_pt(pnl_date_0005_1a['ret_n'], 10, num_per=5000, metric='mean').
        _NOTE_:
        A low p-value (e.g., p<0.05) suggests that the observed performance is statistically significant and likely not due to random chance.
        A high p-value suggests that the observed performance could reasonably have occurred under random shuffling, indicating that the strategy might lack a significant edge.
    """
    returns = dta_in * scale
    if metric == 'mean':
        observed_metric = returns.mean()
    elif metric == 'sharpe':
        observed_metric = returns.mean() / returns.std()
    elif metric == 'cumul':
        observed_metric = np.cumprod(1 + returns).iloc[-1] - 1
    else:
        raise ValueError("Unsupported metric. Use 'mean', 'cumul', or 'sharpe'.")

    permuted_metrics = []
    # loop through 
    for _ in range(num_per):
        # Perform permutations
        permuted_returns = price_permute(returns, 1)
        if metric == 'mean':
            permuted_metric = permuted_returns.mean()
        elif metric == 'sharpe':
            permuted_metric = permuted_returns.mean() / permuted_returns.std()
        elif metric == 'cumul':
            permuted_metric = np.cumprod(1 + permuted_returns).iloc[-1] - 1
        permuted_metrics.append(permuted_metric)

    # Calculate p-value: proportion of permuted metrics >= observed metric
    permuted_metrics = np.array(permuted_metrics)
    p_value = np.mean(permuted_metrics >= observed_metric)
    
    return p_value

# Func 27: Calculate entropy on selected variables in a DataFrame
def entropy_variables(dta_in: pd.DataFrame, nbins: int, var_list: list, min_entr: float = 0.50):
    """
    Calculate entropy on selected variables in a DataFrame.
    - dta_in (pd.DataFrame): Input DataFrame containing variables.
    - nbins (int): Number of bins for histogram calculation.
    - var_list (list): List of variables to calculate entropy for.
    - min_entr (int): Minimum entropy threshold for categorizing variables.
    """
    entr_df = pd.DataFrame()
    results = []

    for var in var_list:
        valid_data = dta_in[var].replace([np.inf, -np.inf], np.nan).dropna()
        if valid_data.empty:
            print(f"Skipping variable '{var}' as it has no valid finite data.")
            continue

        counts, _ = np.histogram(valid_data, bins=nbins)
        if counts.sum() == 0:
            print(f"Skipping variable '{var}' due to zero counts in histogram.")
            continue

        probs = counts / counts.sum()
        if np.isnan(probs).any():
            print(f"Skipping variable '{var}' as it resulted in NaN probabilities.")
            continue

        entros = entropy(probs, base=nbins)
        results.append({'variable': var, 'entropy': entros})

    entr_df = pd.DataFrame(results)
    entr_df.sort_values(by=['entropy'], ascending=False, inplace=True)

    low_entr = entr_df[entr_df['entropy'] < min_entr]['variable'].tolist()
    print(f'Low entropy variables: {len(low_entr)}')
    print(textwrap.fill(", ".join(low_entr), width=250) + "\n")

    high_entr = entr_df[entr_df['entropy'] > min_entr + 0.001]['variable'].tolist()
    print(f'High entropy variables: {len(high_entr)}')
    print(textwrap.fill(", ".join(high_entr), width=250) + "\n")
    
    return entr_df, low_entr, high_entr

def clean_tails(n: int, raw: np.ndarray, tail_frac: float) -> np.ndarray:
    """
    Cleans the tails of the data array `raw` by compressing extreme values based on the desired `tail_frac`.
    
    Parameters:
        n (int): Number of cases.
        raw (np.ndarray): Array of raw data values of length `n`.
        tail_frac (float): Fraction of each tail to be cleaned (between 0 and 0.5).
        
    Returns:
        np.ndarray: The cleaned array with tails compressed, preserving the original order.
    """
    work = np.copy(raw)  # Create a copy to avoid modifying the original data
    cover = 1.0 - 2.0 * tail_frac
    work_sorted = np.sort(work)  # Sort a copy to find boundaries

    # Determine interval size for desired coverage
    istart = 0
    istop = int(cover * (n + 1)) - 1
    if istop >= n:  # Handles cases with tail_frac set to 0
        istop = n - 1

    # Find the narrowest range that covers the desired fraction of data
    best = float("inf")
    best_start = best_stop = 0  # Placeholder indices

    while istop < n:
        range_ = work_sorted[istop] - work_sorted[istart]
        if range_ < best:
            best = range_
            best_start = istart
            best_stop = istop
        istart += 1
        istop += 1

    # Values at the start and end of the best interval
    minval = work_sorted[best_start]
    maxval = work_sorted[best_stop]

    # Handle pathological case where maxval == minval
    if maxval <= minval:
        maxval *= 1.0 + 1e-10
        minval *= 1.0 - 1e-10

    # Scaling factors for compressing tails
    limit = (maxval - minval) * (1.0 - cover)
    scale = -1.0 / (maxval - minval)

    # Apply tail cleaning to values in the original `raw` array (preserves order)
    cleaned = np.copy(raw)  # Copy to preserve original data
    changes_count = 0  # Counter for changes
    for i in range(n):
        if cleaned[i] < minval:  # Left tail
            cleaned[i] = minval - limit * (1.0 - np.exp(scale * (minval - cleaned[i])))
            changes_count += 1
        elif cleaned[i] > maxval:  # Right tail
            cleaned[i] = maxval + limit * (1.0 - np.exp(scale * (cleaned[i] - maxval)))
            changes_count += 1

    print(f"Total changes applied to vector: {changes_count}")
    return cleaned

# Find buckets for Metalabels
def build_meta_sizing_map(
    df: pd.DataFrame,
    prob_col: str = "yhat_train_meta_1",
    pl_col: str = "pl_g",
    date_col: str = "normed_date",   # kept for reference; not required for sorting
    time_col: str = "entry_time",    # full datetime string like "2021-07-15 09:41:01.006"
    win_col: str = "wins",
    n_buckets: int = 5,
    multipliers=(0.0, 0.5, 1.0, 1.5, 2.0),
    min_trades_per_bucket: int = 200,
):
    d = df.copy()
    d = d.dropna(subset=[prob_col, pl_col, time_col]).copy()

    # Robust timestamp parse from entry_time
    d["ts"] = pd.to_datetime(d[time_col].astype(str), errors="coerce", utc=True)
    d = d.dropna(subset=["ts"]).copy()
    d["ts"] = d["ts"].dt.tz_convert(None)  # make naive
    d = d.sort_values("ts").reset_index(drop=True)

    # Ensure wins exists (or recompute)
    if win_col not in d.columns:
        d[win_col] = (d[pl_col] > 0).astype(int)

    # Bucket by probability quantiles (rank-based to avoid duplicated prob edge issues)
    probs = d[prob_col].astype(float)
    ranks = probs.rank(method="first", pct=True)

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    edges[-1] = 1.0000001  # guard
    d["meta_bucket"] = pd.cut(ranks, bins=edges, labels=False, include_lowest=True)

    # Bucket stats
    g = d.groupby("meta_bucket", observed=True)
    bucket_stats = g.agg(
        trades=(pl_col, "size"),
        win_rate=(win_col, "mean"),
        avg_pl=(pl_col, "mean"),
        med_pl=(pl_col, "median"),
        pl_std=(pl_col, "std"),
        sum_pl=(pl_col, "sum"),
        avg_prob=(prob_col, "mean"),
    ).reset_index()

    bucket_stats["avg_pl_tstat"] = bucket_stats["avg_pl"] / (bucket_stats["pl_std"] / np.sqrt(bucket_stats["trades"]))
    bucket_stats["avg_pl_tstat"] = bucket_stats["avg_pl_tstat"].replace([np.inf, -np.inf], np.nan)

    # If buckets are too thin, reduce bucket count automatically (conservative)
    if (bucket_stats["trades"] < min_trades_per_bucket).any() and n_buckets > 2:
        return build_meta_sizing_map(
            df=df,
            prob_col=prob_col,
            pl_col=pl_col,
            date_col=date_col,
            time_col=time_col,
            win_col=win_col,
            n_buckets=n_buckets - 1,
            multipliers=multipliers[: max(2, n_buckets - 1)],
            min_trades_per_bucket=min_trades_per_bucket,
        )

    multipliers = list(multipliers)
    if len(multipliers) != n_buckets:
        raise ValueError(f"multipliers length ({len(multipliers)}) must equal n_buckets ({n_buckets}).")

    sizing_map = {i: float(multipliers[i]) for i in range(n_buckets)}

    d["size_mult"] = d["meta_bucket"].map(sizing_map).astype(float)
    d["sized_pl"] = d[pl_col].astype(float) * d["size_mult"]

    # Add sized performance to bucket_stats
    bucket_stats = bucket_stats.merge(
        d.groupby("meta_bucket", observed=True).agg(
            avg_sized_pl=("sized_pl", "mean"),
            sum_sized_pl=("sized_pl", "sum"),
        ).reset_index(),
        on="meta_bucket",
        how="left",
    )

    return d, bucket_stats.sort_values("meta_bucket"), sizing_map, edges
