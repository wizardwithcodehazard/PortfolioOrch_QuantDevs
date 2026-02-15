# src/ai_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_ai_weights(prices_df, fundamental_df):
    """
    Uses a 'Perceptron-style' optimization to find weights.
    Optimizes the IMPORTANCE of fundamental factors (ROE, PE, Upside).
    """
    # --- FIX START: Clean Data Alignment ---
    # 1. Drop Nifty 50 Index (We don't buy the index, we only buy stocks)
    if '^NSEI' in prices_df.columns:
        clean_prices = prices_df.drop(columns=['^NSEI'])
    else:
        clean_prices = prices_df

    # 2. ALIGNMENT: Ensure Prices columns match Fundamental rows exactly
    # This prevents mismatched matrices (e.g. assigning HAL's risk to SBI)
    clean_prices = clean_prices[fundamental_df.index]
    
    # 3. Calculate Risk (Covariance) on the 8 stocks only
    returns = clean_prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    # --- FIX END ---

    # 4. Prepare Fundamental Features
    # Calculate expected upside from CA's Target Price
    fundamental_df['Upside'] = (fundamental_df['Target_Price'] - fundamental_df['Current_Price']) / fundamental_df['Current_Price']
    
    # Normalize features (Z-score)
    features = fundamental_df[['ROE', 'PE', 'Upside']].copy()
    features = (features - features.mean()) / features.std()
    
    # Feature Matrix (8 Stocks x 3 Features)
    X = features.values 
    
    # 5. Define the Neural Network Logic
    def get_portfolio_weights(theta, feature_matrix):
        # Theta = Weights of the Neural Network
        scores = np.dot(feature_matrix, theta[:3]) + theta[3]
        return softmax(scores) # Softmax ensures weights sum to 1

    # 6. Define Loss Function (Negative Sharpe Ratio)
    def negative_sharpe(theta, feature_matrix, cov_mat):
        weights = get_portfolio_weights(theta, feature_matrix)
        
        # Portfolio Risk (Volatility)
        port_var = np.dot(weights.T, np.dot(cov_mat, weights))
        port_vol = np.sqrt(port_var)
        
        # Portfolio Return (Use CA's 'Upside' forecast as Expected Return)
        exp_ret = np.dot(weights, fundamental_df['Upside'].values)
        
        sharpe = (exp_ret - 0.07) / port_vol # Risk free 7%
        
        # Penalty if any weight > 20% (Competition Constraint)
        penalty = np.sum(np.maximum(0, weights - 0.20)) * 100
        
        return -(sharpe - penalty)

    # 7. Run Optimization with LOGIC BOUNDS
    # Bounds logic: 
    # Theta[0] (ROE):    (0, None)  -> Must be Positive (High ROE is good)
    # Theta[1] (PE):     (None, 0)  -> Must be Negative (Low PE is good)
    # Theta[2] (Upside): (0, None)  -> Must be Positive (High Upside is good)
    # Theta[3] (Bias):   (None, None) -> Can be anything
    
    bounds = [(0, None), (None, 0), (0, None), (None, None)]
    
    # Initial Guess: High ROE (1.0), Low PE (-1.0), High Upside (1.0)
    init_theta = np.array([1.0, -1.0, 1.0, 0.0]) 
    
    result = minimize(
        negative_sharpe, 
        init_theta, 
        args=(X, cov_matrix), 
        method='SLSQP',
        bounds=bounds  # <--- THIS IS THE FIX
    )
    
    # 8. Extract Final Weights
    optimal_theta = result.x
    final_weights = get_portfolio_weights(optimal_theta, X)
    
    weight_dict = dict(zip(fundamental_df.index, final_weights))
    return weight_dict, optimal_theta