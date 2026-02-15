# src/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_monte_carlo(sim_df, save_path="output/monte_carlo.png"):
    """
    Plots the Monte Carlo Fan Chart.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot first 100 paths to show volatility
    plt.plot(sim_df.iloc[:, :100], color='lightgray', alpha=0.5, linewidth=0.5)
    
    # Plot Mean Path
    plt.plot(sim_df.mean(axis=1), color='blue', linewidth=2, label='Mean Expectation')
    
    # Plot 5th and 95th Percentile (Confidence Interval)
    plt.plot(sim_df.quantile(0.05, axis=1), color='red', linestyle='--', label='5th Percentile (Risk)')
    plt.plot(sim_df.quantile(0.95, axis=1), color='green', linestyle='--', label='95th Percentile (Reward)')
    
    plt.title(f"Monte Carlo Simulation: 1-Year Portfolio Forecast (10,000 Runs)")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print(f"📊 Chart saved to {save_path}")
    plt.close()