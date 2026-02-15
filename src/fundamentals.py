# src/fundamentals.py
import yfinance as yf
import pandas as pd

def get_fundamental_scorecard(tickers, manual_data=None):
    """
    Fetches basic fundamentals. 
    manual_data: Dict of dicts for values your CA teammate researched 
                 e.g., {'BEL.NS': {'OrderBook_Revenue': 3.5}}
    """
    scorecard = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            # Fallbacks if yfinance misses data
            roe = info.get('returnOnEquity', 0)
            pe = info.get('trailingPE', 0)
            peg = info.get('pegRatio', 0)
            debt_eq = info.get('debtToEquity', 0) / 100 # yf gives it as percentage
            
            # Custom Logic: "Safe" if Debt/Eq < 1 and ROE > 0.15
            status = "Pass" if (debt_eq < 1.0 and roe > 0.15) else "Check"
            
        except Exception as e:
            print(f"⚠️ Could not fetch fundamentals for {ticker}: {e}")
            roe, pe, peg, debt_eq, status = 0, 0, 0, 0, "Error"

        # Override with CA's manual research if provided
        ca_note = ""
        if manual_data and ticker in manual_data:
            ca_note = manual_data[ticker].get("Note", "")

        scorecard.append({
            "Ticker": ticker,
            "ROE": round(roe, 2),
            "PE": round(pe, 2),
            "PEG": round(peg, 2),
            "Debt_Equity": round(debt_eq, 2),
            "Quant_Status": status,
            "CA_Note": ca_note
        })
        
    return pd.DataFrame(scorecard).set_index("Ticker")