# src/optimizer.py
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_portfolio(prices_df):
    """
    Optimizes for Max Sharpe Ratio with a constraint: Max weight = 20%.
    """
    # Clean data: drop Nifty 50 for portfolio construction
    if '^NSEI' in prices_df.columns:
        df = prices_df.drop(columns=['^NSEI'])
    else:
        df = prices_df

    # 1. Calculate Expected Returns and Sample Covariance
    mu = expected_returns.capm_return(df)
    S = risk_models.sample_cov(df)

    # 2. Optimize
    ef = EfficientFrontier(mu, S)
    
    # CONSTRAINT: No single asset > 20% (0.2)
    # We also assume no shorting (weights range 0 to 1)
    ef.add_constraint(lambda x: x <= 0.20)
    
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    performance = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, performance, ef