# src/simulator.py
import numpy as np
import pandas as pd

def run_monte_carlo(prices_df, weights, simulations=10000, days=252):
    """
    Runs Monte Carlo simulation for the portfolio.
    Returns the simulation DataFrame.
    """
    # Drop Nifty
    if '^NSEI' in prices_df.columns:
        prices = prices_df.drop(columns=['^NSEI'])
    else:
        prices = prices_df
        
    log_returns = np.log(1 + prices.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    
    # drift = u - (0.5 * var)
    drift = u.values
    stdev = log_returns.std().values
    
    # Portfolio Weights array
    weights_arr = np.array(list(weights.values()))
    
    # Simulation storage
    portfolio_sims = np.zeros((days, simulations))
    
    # Initial Portfolio Value
    initial_value = 10000000 # 1 Crore
    
    for sim in range(simulations):
        # Generate random daily returns for all stocks
        # Shape: (days, num_stocks)
        Z = np.random.normal(0, 1, (days, len(weights)))
        daily_returns = np.exp(drift + stdev * Z)
        
        # Calculate portfolio path
        # We simplify by assuming rebalancing isn't daily, but for a 1-year view, 
        # we treat the portfolio as a composite asset.
        
        # Simulating the weighted portfolio return path
        port_daily_ret = np.dot(daily_returns, weights_arr)
        
        # Accumulate returns
        price_path = np.zeros(days)
        price_path[0] = initial_value * port_daily_ret[0]
        
        for t in range(1, days):
            price_path[t] = price_path[t-1] * port_daily_ret[t]
            
        portfolio_sims[:, sim] = price_path
        
    return pd.DataFrame(portfolio_sims)