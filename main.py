"""
Indian Market Trading Agent — Short-term Trading Decision System
Based on TradingAgents framework, adapted for NSE/BSE markets.
"""

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.utils.ticker import normalize_ticker

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Create config for Indian market short-term trading
config = DEFAULT_CONFIG.copy()

# LLM settings — Anthropic Claude
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-3.5-flash"
config["quick_think_llm"] = "gemini-3.5-flash"
config["google_thinking_level"] = "high"

# Debate settings
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

# Data vendors (yfinance for now, Kite API in Phase 4)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
    "indian_market_data": "nse",
}

# Initialize the trading agent graph
ta = TradingAgentsGraph(debug=True, config=config)

# --- Example: Analyze an Indian stock ---
# Use NSE tickers: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, etc.
# For indices: ^NSEI (NIFTY 50), ^NSEBANK (BANK NIFTY)
ticker = normalize_ticker("RELIANCE")  # Automatically adds .NS suffix
trade_date = "2025-04-10"

print(f"\n{'='*60}")
print(f"  Indian Market Trading Agent")
print(f"  Analyzing: {ticker}")
print(f"  Date: {trade_date}")
print(f"  Style: Short-term trading")
print(f"{'='*60}\n")

# Run the full multi-agent analysis pipeline
_, decision = ta.propagate(ticker, trade_date)
print("\n" + "="*60)
print("FINAL TRADING DECISION:")
print("="*60)
print(decision)

# Uncomment to enable memory/learning after tracking trade performance:
# ta.reflect_and_remember(returns_in_inr)  # Pass actual P&L to learn from trades
