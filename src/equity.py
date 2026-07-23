import yfinance as yf
import pandas as pd
import io

# 1. Your initial data (The "Ground Truth" from your CA Report)
csv_data = """Ticker,Sector,Current_Price,Target_Price,ROE,PE,Growth_Forecast,CA_Note
ICICIBANK.NS,Banking,1406.65,1717.11,18.98,20.05,22.07,Highest operational efficiency and digital lead.
SBIN.NS,Banking,1195.70,1206.41,16.00,13.50,0.90,Largest scale; reliability and dividend play.
HAL.NS,Defence,4212.40,5323.05,26.09,33.68,26.37,Monopoly in aerospace; record order book.
BEL.NS,Defence,435.55,525.00,29.29,53.38,20.54,Dominant in defence electronics; high ROCE.
SUNPHARMA.NS,Pharma,1622.60,1981.90,16.50,34.00,22.14,Specialty focus; beneficiary of Biopharma Shakti.
BIOCON.NS,Pharma,377.95,425.37,11.00,30.00,12.55,Pure play on biologics; BBL integration.
TATAPOWER.NS,Power,374.10,468.25,12.90,28.00,25.17,Integrated green energy leader; EV infra.
POWERGRID.NS,Power,294.00,315.19,7.27,12.00,7.21,Transmission monopoly; stable capex-led growth."""

def update_stock_universe(csv_input):
    # Load into DataFrame
    df = pd.read_csv(io.StringIO(csv_input))
    
    print("🚀 Fetching live market data from Yahoo Finance...")
    
    for index, row in df.iterrows():
        ticker_symbol = row['Ticker']
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # 2. Update Current Price
            # Try multiple price keys as yfinance can be inconsistent between markets
            live_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
            if live_price:
                df.at[index, 'Current_Price'] = round(live_price, 2)
            
            # 3. Update PE Ratio (Trailing)
            pe_ratio = info.get('trailingPE')
            if pe_ratio:
                df.at[index, 'PE'] = round(pe_ratio, 2)
                
            # 4. Update ROE
            # yfinance returns ROE as a decimal (0.18 for 18%), so we multiply by 100
            roe = info.get('returnOnEquity')
            if roe:
                df.at[index, 'ROE'] = round(roe * 100, 2)
            
            print(f"✅ Updated {ticker_symbol}")
            
        except Exception as e:
            print(f"⚠️ Could not update {ticker_symbol}: {e}")

    # 5. Save the updated data
    df.to_csv("updated_stock_universe.csv", index=False)
    print("\n✨ Update complete! Saved to 'updated_stock_universe.csv'")
    return df

if __name__ == "__main__":
    updated_df = update_stock_universe(csv_data)
    print("\n--- UPDATED PREVIEW ---")
    print(updated_df[['Ticker', 'Current_Price', 'ROE', 'PE']])