# src/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

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