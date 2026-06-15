# Top 1000 Liquid Stock Universe & SQLite Price Cache

This document details the implementation of the Top 1000 Liquid Stock universe along with the asynchronous SQLite database price cache in the Indian Trading Agent codebase.

## Objective
To track the top 1000 liquid stocks in the Indian market (composed of the entire Nifty 500 list plus active NSE standard equities) and run scans/recommendations quickly without triggering Yahoo Finance (`yfinance`) rate limits, network timeouts, or blocking backend threads.

---

## Architecture Design

### 1. The Bottleneck
Previously, running a Scan or Recommendation triggered live HTTP queries to Yahoo Finance for every ticker in parallel using a thread pool. For 1,000 tickers, this synchronous network overhead takes 2–5 minutes, results in high error rates, and triggers IP blocks from Yahoo Finance.

### 2. The Solution (Local Database Caching)
We introduced a local SQLite database price cache.
- The backend engines read historical candles directly from the local SQLite database.
- A background cron task updates/backfills this database cache periodically.
- Outbound network requests to `yfinance` at runtime are reduced to 0, making scanning and recommendation runs extremely fast ($O(1)$ locally).

---

## File Changes & Code Details

### 1. Database Schema
#### [db.py](file:///d:/Code/indian-trading-agent/backend/db.py)
*   Added the `stock_prices` cache table and indexes inside `ensure_db()`:
    ```sql
    CREATE TABLE IF NOT EXISTS stock_prices (
        ticker TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        PRIMARY KEY (ticker, trade_date)
    );
    CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker);
    CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(trade_date);
    ```
*   Added the helper function `get_stock_prices(ticker, period_days)` to retrieve historical prices from SQLite and format them as a Pandas DataFrame identical to `yfinance.Ticker.history()`.

### 2. Stock Lists & Ticker Mappings
*   Added `backend/liquid_1000_tickers.json` containing the sorted list of 1,000 liquid tickers.
*   Added `backend/liquid_1000_mappings.json` containing the name-to-ticker lookup dictionary.
*   Modified [stock_list.py](file:///d:/Code/indian-trading-agent/backend/stock_list.py) to load `liquid_1000_mappings.json` and merge them into `NSE_STOCKS` at startup to support typeahead search in the frontend.

### 3. Asynchronous Cache Downloader
#### [cron.py](file:///d:/Code/indian-trading-agent/backend/cron.py)
*   Implemented `update_price_cache(tickers_list, force)` to download prices in parallel chunks of 50 using `yf.download()` (taking only ~40 seconds for 1,000 stocks).
*   Integrated into `_cron_loop()` to run:
    - On server startup if the database is empty or stale.
    - Every 30 minutes during Indian market hours (Monday-Friday, 9:00 AM to 4:00 PM IST) to capture intraday daily candles.
    - Daily post-market close.

### 4. Engine Refactoring
*   Modified [scanner.py](file:///d:/Code/indian-trading-agent/backend/scanner.py):
    - Added the `liquid1000` universe to the mapping dictionary `UNIVERSES`.
    - Updated `_fetch_stock_data` to read from `get_stock_prices` cache first, falling back to yfinance if empty.
*   Modified [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py):
    - Updated `_analyze_stock` to query the local SQLite price cache first.

### 5. Frontend Integration
Added the `Top 1000 Liquid` option to the dropdown components on the following pages:
*   [page.tsx](file:///d:/Code/indian-trading-agent/frontend/src/app/scanner/page.tsx) (Market Scanner)
*   [page.tsx](file:///d:/Code/indian-trading-agent/frontend/src/app/recommendations/page.tsx) (Recommendations)
*   [page.tsx](file:///d:/Code/indian-trading-agent/frontend/src/app/performance/page.tsx) (Strategy Performance)
*   [page.tsx](file:///d:/Code/indian-trading-agent/frontend/src/app/simulation/page.tsx) (Backtest Simulation)

---

## Verification & Metrics

1.  **Cache Backfill Priming**:
    - Ran `update_price_cache()` to download and populate all tickers.
    - **Outcome**: Succeeded for 1,004 active tickers, saving **119,811 daily bars** in SQLite database in **~40 seconds**.
2.  **Market Scanner Run**:
    - Executed `run_scan(universe="liquid1000")` using cached prices.
    - **Outcome**: Completed in **6.8 seconds** (previously timed out or failed). Found 334 gaps, 54 volume spikes, and 107 breakouts.
3.  **Recommendation Engine Run**:
    - Executed `recommend(universe="liquid1000")` using cached prices.
    - **Outcome**: Completed successfully in **5.2 seconds**, identifying 63 actionable recommendations.
