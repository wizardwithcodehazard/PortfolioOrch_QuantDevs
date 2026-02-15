# src/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

def plot_monte_carlo(sim_df, save_path="output/monte_carlo.png"):
    """
    Plots a professional 'Hedge Fund Style' Monte Carlo chart.
    """
    # Set sophisticated style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Plot the "Chaos" (Background Paths)
    # Plot fewer paths (500) to keep it clean, but make them subtle
    ax.plot(sim_df.iloc[:, :500], color='lightgray', alpha=0.15, linewidth=0.5, zorder=1)
    
    # 2. Plot the "Mean Expectation" (The Blue Line)
    mean_path = sim_df.mean(axis=1)
    ax.plot(mean_path, color='#004aad', linewidth=3, label='Expected Path (Mean)', zorder=3)
    
    # 3. Plot Risk Bounds (Red/Green)
    worst_case = sim_df.quantile(0.05, axis=1)
    best_case = sim_df.quantile(0.95, axis=1)
    
    ax.plot(worst_case, color='#d62728', linestyle='--', linewidth=2, label='Worst Case (95% Confidence)', zorder=3)
    ax.plot(best_case, color='#2ca02c', linestyle='--', linewidth=2, label='Best Case (5% Confidence)', zorder=3)
    
    # 4. The "Break-Even" Line (Crucial for Psychology)
    initial_capital = 10000000
    ax.axhline(y=initial_capital, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial Capital (₹1 Cr)', zorder=2)
    
    # 5. Smart Annotations (The "Callouts")
    # Get final values
    final_mean = mean_path.iloc[-1]
    final_worst = worst_case.iloc[-1]
    
    # Add text bubbles at the end of the lines
    ax.annotate(f'₹{final_mean/1e7:.2f} Cr\n(+{(final_mean/initial_capital - 1)*100:.1f}%)', 
                xy=(252, final_mean), 
                xytext=(260, final_mean),
                arrowprops=dict(arrowstyle="-", color='#004aad'),
                color='#004aad', fontweight='bold', fontsize=12, va='center')

    ax.annotate(f'₹{final_worst/1e7:.2f} Cr', 
                xy=(252, final_worst), 
                xytext=(260, final_worst),
                arrowprops=dict(arrowstyle="-", color='#d62728'),
                color='#d62728', fontsize=10, va='center')

    # 6. Format Axis to "Crores"
    def crores_fmt(x, pos):
        return f'₹{x/1e7:.2f} Cr'
    
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(crores_fmt))
    
    # Labels & Titles
    plt.title("Monte Carlo Forecast: 1-Year Risk/Reward Analysis", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Trading Days (2026)", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    
    # Legend
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True, fontsize=10)
    
    # Adjust layout to make room for annotations
    plt.subplots_adjust(right=0.85)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Professional Chart saved to {save_path}")
    plt.close()

def plot_correlation_matrix(prices_df, save_path="output/correlation_matrix.png"):
    """
    Plots a correlation heatmap of the stock returns. 
    Using Matplotlib to avoid extra dependencies (Seaborn).
    """
    # Calculate Log Returns
    if '^NSEI' in prices_df.columns:
        prices = prices_df.drop(columns=['^NSEI'])
    else:
        prices = prices_df
        
    returns = np.log(1 + prices.pct_change().dropna())
    corr_matrix = returns.corr()
    
    # Setup Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Heatmap
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add Colorbar
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    # Add Text Annotations
    rows, cols = corr_matrix.shape
    for i in range(rows):
        for j in range(cols):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, '{:0.2f}'.format(val), ha='center', va='center', 
                    color='white' if abs(val) > 0.5 else 'black', fontsize=10)
    
    # Labels
    tickers = corr_matrix.columns
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='left')
    ax.set_yticklabels(tickers)
    
    ax.set_title("Asset Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Correlation Matrix saved to {save_path}")
    plt.close(fig)

def plot_efficient_frontier(prices_df, num_portfolios=2000, save_path="output/efficient_frontier.png"):
    """
    Simulates random portfolios to visualize the Efficient Frontier.
    Plots Volatility vs. Expected Return.
    """
    if '^NSEI' in prices_df.columns:
        prices = prices_df.drop(columns=['^NSEI'])
    else:
        prices = prices_df

    # Annualized Returns & Covariance
    returns = np.log(1 + prices.pct_change().dropna())
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Portfolio Weights Simulation
    port_returns = []
    port_volatilities = []
    port_sharpe_ratios = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(prices.columns))
        weights /= np.sum(weights)
        
        # Returns
        ret = np.sum(mean_daily_returns * weights) * 252
        port_returns.append(ret)
        
        # Volatility
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        port_volatilities.append(vol)
        
        # Sharpe (Rf = 0.07)
        sharpe = (ret - 0.07) / vol
        port_sharpe_ratios.append(sharpe)
    
    # Create DataFrame
    sim_df = pd.DataFrame({
        'Return': port_returns, 
        'Volatility': port_volatilities, 
        'Sharpe': port_sharpe_ratios
    })
    
    # Max Sharpe
    max_sharpe = sim_df.iloc[sim_df['Sharpe'].idxmax()]
    min_vol = sim_df.iloc[sim_df['Volatility'].idxmin()]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sim_df['Volatility'], sim_df['Return'], c=sim_df['Sharpe'], cmap='viridis', alpha=0.5, s=10)
    fig.colorbar(sc, label='Sharpe Ratio')
    
    # Highlight
    ax.scatter(max_sharpe['Volatility'], max_sharpe['Return'], c='red', s=100, marker='*', label='Max Sharpe Ratio')
    ax.scatter(min_vol['Volatility'], min_vol['Return'], c='blue', s=100, marker='*', label='Min Volatility')
    
    ax.set_title('Efficient Frontier Simulation', fontsize=16, fontweight='bold')
    ax.set_xlabel('Annualized Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return', fontsize=12)
    ax.legend()
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Efficient Frontier saved to {save_path}")
    except Exception as e:
        print(f"❌ FAILED to save Efficient Frontier: {e}")
        
    plt.close(fig)

def plot_asset_allocation(weights_dict, save_path="output/asset_allocation.png"):
    """
    Plots a donut chart of the portfolio weights.
    """
    # Filter
    labels = []
    sizes = []
    for t, w in weights_dict.items():
        if w > 0.01:
            labels.append(t)
            sizes.append(w)
            
    # Normalize
    sizes = np.array(sizes)
    sizes = sizes / sizes.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.85,
                                    wedgeprops=dict(width=0.4, edgecolor='w'))
                                    
    plt.setp(texts, size=12, weight="bold")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    
    ax.set_title("Optimized Asset Allocation", fontsize=16, fontweight='bold')
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Asset Allocation saved to {save_path}")
    plt.close(fig)