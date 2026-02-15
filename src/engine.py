# src/engine.py
import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.07  # 7% for India 1Y T-Bill
MARKET_RETURN = 0.13   # 13% Nifty Historical Avg

def calculate_metrics(prices_df):
    """
    Returns a DataFrame with Beta, Volatility, and Expected Return (CAPM).
    """
    metrics = []
    
    # Calculate daily returns
    returns = prices_df.pct_change().dropna()
    market_returns = returns['^NSEI']
    
    for ticker in prices_df.columns:
        if ticker == '^NSEI':
            continue
            
        stock_returns = returns[ticker]
        
        # 1. Beta Calculation (Covariance / Variance)
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance
        
        # 2. Annualized Volatility (Standard Deviation * sqrt(252))
        volatility = stock_returns.std() * np.sqrt(252)
        
        # 3. CAPM Expected Return
        expected_return = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)
        
        # 4. Moving Average Check (Trend)
        current_price = prices_df[ticker].iloc[-1]
        ma_200 = prices_df[ticker].rolling(window=200).mean().iloc[-1]
        trend = "UP" if current_price > ma_200 else "DOWN"

        metrics.append({
            "Ticker": ticker,
            "Beta": round(beta, 2),
            "Volatility_1Y": round(volatility, 2),
            "Exp_Return_CAPM": round(expected_return, 2),
            "Trend_200DMA": trend,
            "Current_Price": round(current_price, 2)
        })
        
    return pd.DataFrame(metrics).set_index("Ticker")