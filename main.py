# main.py
import typer
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.status import Status
from src import fetcher, engine, ai_optimizer, simulator, visualizer, verifier

# Initialize Typer and Rich Console
app = typer.Typer()
console = Console()

# Correct Tickers from CA Report
TICKERS = [
    "ICICIBANK.NS", "SBIN.NS",      # Banking
    "HAL.NS", "BEL.NS",             # Defence
    "SUNPHARMA.NS", "BIOCON.NS",    # Pharma
    "TATAPOWER.NS", "POWERGRID.NS"  # Power
]

@app.command()
def run_analysis():
    """
    Run the full CBS Finance Challenge AI-Quant Pipeline.
    """
    console.print(Panel.fit(
        "[bold cyan]🚀 CBS Finance Challenge: AI-Quant Pipeline[/bold cyan]\n"
        "[dim]Strategy over Speculation | 2026 Portfolio Audit[/dim]",
        border_style="cyan"
    ))

    # 1. Load CA Data
    with console.status("[bold yellow]Loading CA Research Data...[/bold yellow]"):
        try:
            fund_df = pd.read_csv("data/stock_universe.csv").set_index("Ticker")
            console.print("✅ [green]CA Research Loaded from CSV.[/green]")
        except FileNotFoundError:
            console.print("❌ [red]Error: Create 'data/stock_universe.csv' first.[/red]")
            raise typer.Exit()

    # 2. Fetch Market Data
    with console.status("[bold yellow]Fetching Market Data & Calculating Metrics...[/bold yellow]"):
        prices = fetcher.fetch_data(TICKERS)
        console.print("✅ [green]Market Data Cached and Synced.[/green]")

    # 3. AI Optimization
    with console.status("[bold magenta]Training Neural Network for Optimal Weights...[/bold magenta]"):
        weights, theta = ai_optimizer.get_ai_weights(prices, fund_df)
        
        table = Table(title="AI-Optimized Weights", border_style="magenta")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Allocation (%)", justify="right", style="green")
        table.add_column("Feature Influence", style="dim")

        # Map theta importance for display
        importance = f"ROE: {theta[0]:.2f} | PE: {theta[1]:.2f} | Upside: {theta[2]:.2f}"
        
        for ticker, weight in weights.items():
            if weight > 0.005:
                table.add_row(ticker, f"{weight*100:.2f}%", importance if ticker == TICKERS[0] else "")
        
        console.print(table)

    # 4. MCMC Simulation
    with console.status("[bold blue]Running Markov Chain Monte Carlo (10,000 Iterations)...[/bold blue]"):
        sim_df = simulator.run_mcmc_simulation(prices, weights, fund_df=fund_df)
        
        final_mean = sim_df.iloc[-1].mean()
        worst_case = sim_df.iloc[-1].quantile(0.05)

        console.print(Panel(
            f"[bold white]Expected Value (1 Yr):[/bold white] [green]₹{final_mean:,.2f}[/green]\n"
            f"[bold white]Worst Case (95% Conf):[/bold white] [red]₹{worst_case:,.2f}[/red]",
            title="[bold blue]Risk Analysis (MCMC)[/bold blue]",
            expand=False
        ))

    # 5. Real-Time Verification
    with console.status("[bold green]Performing Groq AI Real-Time Browser Audit...[/bold green]"):
        live_audit = verifier.verify_with_groq(weights, fund_df)
    
    console.print("\n[bold cyan]─── FINAL INVESTMENT INFERENCE (GROQ AI) ───[/bold cyan]")
    # Render the Groq AI output as Markdown for a professional look
    console.print(Markdown(live_audit))

    # 6. Finalize Visualization
    with console.status("[bold white]Generating Professional Charts...[/bold white]"):
        visualizer.plot_monte_carlo(sim_df)
    
    console.print(f"\n[bold green]✨ Pipeline Complete.[/bold green] Charts saved to [dim]output/monte_carlo.png[/dim]")

if __name__ == "__main__":
    app()