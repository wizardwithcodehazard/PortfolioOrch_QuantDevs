# src/fetcher.py
import yfinance as yf
import pandas as pd
import os

DATA_DIR = "data"

def fetch_data(tickers, start_date="2024-01-01", end_date="2026-02-15"):
    """
    Fetches adjusted close prices for tickers + Nifty 50 (^NSEI).
    Caches data to CSV to avoid repeated API calls.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Add Nifty 50 for Beta calculations
    all_tickers = tickers + ['^NSEI']
    file_path = os.path.join(DATA_DIR, "market_data.csv")
    
    if os.path.exists(file_path):
        print(f"📦 Loading data from local cache: {file_path}")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"⬇️ Downloading data for {len(all_tickers)} tickers...")
        
        # FIX: auto_adjust=False guarantees 'Adj Close' exists
        # multi_level_index=False ensures we don't get nested columns
        data_raw = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=False)
        
        # Handle different yfinance versions (MultiIndex vs Single Index)
        if isinstance(data_raw.columns, pd.MultiIndex):
            # Check if 'Adj Close' is in the top level
            if 'Adj Close' in data_raw.columns.get_level_values(0):
                data = data_raw['Adj Close']
            else:
                # Fallback to 'Close' if Adj Close is missing
                print("⚠️ 'Adj Close' not found, using 'Close'...")
                data = data_raw['Close']
        else:
            data = data_raw['Adj Close']

        # Save to CSV
        data.to_csv(file_path)
        print("✅ Data saved to cache.")
    
    # Drop columns that are completely empty (failed downloads)
    data.dropna(axis=1, how='all', inplace=True)
    
    # Validation: Ensure ^NSEI exists
    if '^NSEI' not in data.columns:
        raise KeyError("❌ Critical Error: Nifty 50 (^NSEI) failed to download. Check your internet or try again later.")

    return data