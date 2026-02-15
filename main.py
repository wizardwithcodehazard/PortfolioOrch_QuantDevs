# main.py
import pandas as pd
from colorama import Fore, Style
from src import fetcher, engine, fundamentals, optimizer, simulator, visualizer

# === INPUTS ===
# 8 Stocks from 4 Industries (Example List)
# main.py (Update the TICKERS list)
TICKERS = [
    "BEL.NS", "HAL.NS",          # Defence
    "HDFCBANK.NS", "ICICIBANK.NS", # Banking
    "TATASTEEL.NS", "JSL.NS",    # Metal (Fixed Spelling: TATASTEEL)
    "TATAPOWER.NS", "NTPC.NS"    # Power
]

# CA's Manual Notes (Optional)
CA_NOTES = {
    "BEL.NS": {"Note": "Order book 3x revenue, Defence tailwind."},
    "HDFCBANK.NS": {"Note": "Merger synergies kicking in 2026."}
}

def main():
    print(f"{Fore.CYAN}🚀 Starting CBS Finance Challenge Portfolio Tool...{Style.RESET_ALL}")
    
    # 1. Fetch Data
    prices = fetcher.fetch_data(TICKERS)
    
    # 2. Run Engine (Math Tests)
    print(f"\n{Fore.YELLOW}⚙️  Running Quant Engine...{Style.RESET_ALL}")
    quant_metrics = engine.calculate_metrics(prices)
    print(quant_metrics[['Beta', 'Exp_Return_CAPM', 'Trend_200DMA']])
    
    # 3. Run Fundamental Check
    print(f"\n{Fore.YELLOW}📜 Running Fundamental Scorecard...{Style.RESET_ALL}")
    fund_metrics = fundamentals.get_fundamental_scorecard(TICKERS, CA_NOTES)
    print(fund_metrics[['ROE', 'Quant_Status', 'CA_Note']])
    
    # 4. Optimize Weights
    print(f"\n{Fore.YELLOW}⚖️  Optimizing Portfolio (Max Sharpe, <20% Weight)...{Style.RESET_ALL}")
    weights, perf, ef = optimizer.optimize_portfolio(prices)
    print("✅ Optimized Weights:")
    for ticker, weight in weights.items():
        if weight > 0:
            print(f"   {ticker}: {round(weight*100, 2)}%")
            
    print(f"\nExpected Annual Return: {perf[0]:.2%}")
    print(f"Annual Volatility: {perf[1]:.2%}")
    print(f"Sharpe Ratio: {perf[2]:.2f}")
    
    # 5. Monte Carlo Simulation
    print(f"\n{Fore.YELLOW}🎲 Running Monte Carlo Simulation (10k Runs)...{Style.RESET_ALL}")
    sim_df = simulator.run_monte_carlo(prices, weights)
    
    final_mean = sim_df.iloc[-1].mean()
    worst_case = sim_df.iloc[-1].quantile(0.05)
    
    print(f"   Expected Value (1 Yr): ₹{final_mean:,.2f}")
    print(f"   Worst Case (95% Conf): ₹{worst_case:,.2f}")
    
    # 6. Visualize
    visualizer.plot_monte_carlo(sim_df)

if __name__ == "__main__":
    main()