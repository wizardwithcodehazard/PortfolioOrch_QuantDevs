# src/simulator.py
import numpy as np
import pandas as pd

def run_mcmc_simulation(prices_df, weights_dict, fund_df=None, simulations=10000, days=252):
    """
    Runs a Markov Chain Monte Carlo (MCMC) style simulation using 
    Historical Bootstrapping with 'Black-Litterman' style Drift Adjustment.
    
    Logic:
    1. RISK: Sample volatility/crashes from REAL History (Fat Tails).
    2. RETURN: Shift the mean trend to match CA's 'Target Price' Forecasts.
    """
    # 1. Prepare Price Data
    if '^NSEI' in prices_df.columns:
        prices = prices_df.drop(columns=['^NSEI'])
    else:
        prices = prices_df
        
    # Align columns
    tickers = list(prices.columns)

    # 2. Calculate Historical Daily Log Returns
    returns = np.log(1 + prices.pct_change().dropna())
    
    # 3. Construct Portfolio Historical Returns
    # Ensure weights align exactly with price columns
    weight_arr = np.array([weights_dict.get(t, 0) for t in tickers])
    port_hist_returns = np.dot(returns.values, weight_arr)
    
    # 4. Calculate "Target Drift" (The Analyst View)
    if fund_df is not None:
        # Ensure 'Upside' column exists
        if 'Upside' not in fund_df.columns:
             fund_df['Upside'] = (fund_df['Target_Price'] - fund_df['Current_Price']) / fund_df['Current_Price']
        
        # Calculate Expected Annual Return based on CA's targets & Weights
        # We look up the upside for the specific tickers in the price columns
        upsides = fund_df.loc[tickers]['Upside'].values
        target_annual_return = np.dot(upsides, weight_arr)
        
        # Convert Annual Target to Daily Log Return Mean
        # Formula: ln(1 + Annual_Ret) / 252
        target_daily_mean = np.log(1 + target_annual_return) / 252
    else:
        # Fallback: Use historical mean (Bearish if history was bad)
        target_daily_mean = port_hist_returns.mean()

    # 5. Calculate Drift Adjustment Factor
    # We want samples to center around 'target_daily_mean', not 'hist_mean'
    hist_mean = port_hist_returns.mean()
    drift_adjustment = target_daily_mean - hist_mean

    # 6. MCMC / Bootstrap Engine
    # Generate random indices to sample from history
    random_indices = np.random.choice(len(port_hist_returns), size=(days, simulations))
    
    # Sampled Returns + Drift Adjustment
    # This preserves historical Volatility (Shape) but fixes the Mean (Trend)
    simulated_daily_returns = port_hist_returns[random_indices] + drift_adjustment

    # 7. Construct Price Paths
    initial_capital = 10000000 # 1 Crore
    
    # Cumulative Return Path: (1+r1)*(1+r2)...
    cum_returns = np.exp(np.cumsum(simulated_daily_returns, axis=0))
    
    price_paths = initial_capital * cum_returns
    
    # Add starting row of 1 Crore
    start_row = np.full((1, simulations), initial_capital)
    full_paths = np.vstack([start_row, price_paths])
    
    return pd.DataFrame(full_paths)