# src/verifier.py
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def verify_with_groq(portfolio_weights, fund_df, max_retries=3):
    """
    Thematic Groq Browser Search auditor. 
    Verifies the portfolio against sector-specific catalysts from the CA report.
    """
    # Initialize client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # 1. Dynamically identify all industries in the portfolio 
    sectors = fund_df['Sector'].unique().tolist()
    
    # Filter for core holdings (>5%) to optimize token usage
    top_stocks = [k for k, v in portfolio_weights.items() if v > 0.05]
    stock_list = ", ".join(top_stocks)
    
    # 2. Refined Thematic Prompt based on the CA's 2026 Thesis 
    prompt = f"""
    Act as a Senior Institutional Strategist. Perform a real-time 'Live Audit' of our 2026 India Portfolio: {stock_list}.
    
    Sectors to Audit: {', '.join(sectors)}.
    
    CRITICAL TASKS:
    1. Scan for regulatory 'Red Flags' (RBI, SEBI, Ministry of Defence, FDA) in the last 72h.
    2. Sentiment Audit: Verify if live news supports these 2026 structural themes:
       - Banking: 'Viksit Bharat' credit growth and ICICI's digital lead.
       - Defence: 'Atmanirbhar Bharat' momentum and HAL/BEL order book visibility.
       - Pharma: 'Biopharma Shakti' initiative impact and biologics approvals.
       - Power: 'Green Energy Corridors' and 'PM Surya Ghar' scheme updates.
    3. Final Verdict: Provide a concise 'Go/No-Go' for each sector based on latest data.
    """

    print(f"🌐 Performing Thematic Audit on {len(sectors)} sectors via Groq Browser Search...")
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.5, # Lowered for higher factual accuracy
                max_completion_tokens=2048, 
                reasoning_effort="low", # Best practice for token efficiency
                tool_choice="required",
                tools=[{"type": "browser_search"}]
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 30 
                print(f"⚠️ Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                return f"⚠️ Unexpected Groq Error: {str(e)}"
    
    return "❌ Max retries reached. Groq is currently throttled."