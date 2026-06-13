# Engineering Logs & Guardrails

To prevent repeating the same bugs and architectural mistakes, here is the list of identified anti-patterns and engineering rules derived from past fixes.

## 🛠️ The Trading System Engineering Playbook

### 1. Market Calendar vs. Gregorian Calendar
*   **The Anti-Pattern**: Advancing historical backtests or expiry dates using calendar-day logic (`timedelta(days=1)`) or standard weekday checks. This skips market holidays (e.g. Diwali, NSE holidays) and introduces weekend skew/Monday lock-in.
*   **The Rule**: Never use raw day-arithmetic for trade/backtest advancement or expiration checks. Always use calendar utilities (`next_trading_day`, `count_trading_days`) backed by actual exchange calendars.

### 2. State & Filter Desynchronization
*   **The Anti-Pattern**: Modifying signal scores or directions in post-processing filters (FII flows, concentration limits, macro events) without atomically updating confidence, counts, stop-loss/targets, or trade plan profiles.
*   **The Rule**: Keep base technical signals and risk-mitigation filters decoupled. Run assessment/trade-plan recalculation as a single, atomic post-filter step (`_recompute_assessment_and_trade_plan()`) so that downstream systems (cron, database caching, UI conviction badges) never use mismatched or pre-filter metrics.

### 3. Data-Tier Safety & Sanitization
*   **The Anti-Pattern**: Duplicate trade insertion across paper/shadow caches, `ZeroDivisionError` on strategies with perfect win/loss records (no losses), and simulations defaulting or crashing on `NULL` database column values.
*   **The Rule**:
    *   Enforce deterministic Python-tier deduplication on composite keys like `(ticker, entry_date, fingerprint)`.
    *   Guard all math formulas (e.g., Kelly Criterion, win rates) against boundary conditions (zero losses, zero wins, empty datasets).
    *   Provide explicit safe fallbacks for `NULL` fields in historical records during simulation passes.

### 4. Price-Level Ingestion (Intraday vs. EOD)
*   **The Anti-Pattern**: Using closing prices (`current_close`) for gap detection or backtest triggers, which misses intraday fill-and-reversal behavior and reports incorrect outcomes.
*   **The Rule**: Intraday logic must check extreme levels (`current_high`/`current_low`) rather than EOD closes to determine gap fill/fade status and stop-loss hits.

### 5. Compounding & Drawdown Realism
*   **The Anti-Pattern**: Understating drawdowns by using static cash sizes, neglecting marked-to-market unrealized P&L, sorting entry after exit on same-day trades, or omitting hit stop-losses.
*   **The Rule**: Simulations must emulate exact exchange logistics: sort same-day trades (entry before exit), include live mark-to-market unrealized P&L, handle stop-losses, and fetch fresh quotes for active positions.

### 6. Query Hygiene & Performance
*   **The Anti-Pattern**: Executing queries in loops (N+1 query problem) when fetching trade stats, causing performance degradation under load, and swallowing exceptions silently.
*   **The Rule**: Batch database operations using `IN` clauses, implement in-memory caching for repetitive computations, and always use structured logging to surface errors rather than swallowing them.

### 7. Technical Indicator Fidelity (Volatility Calculations)
*   **The Anti-Pattern**: Mixing indicator definitions or using simple arithmetic means (SMA) where Wilder's smoothing/exponential smoothing is mathematically specified and expected (e.g. for Average True Range). This underestimates short-term volatility spikes, leading to overly tight stop-losses and premature stop-outs.
*   **The Rule**: Ensure mathematical implementations of indicators strictly match the industry standards described in their documentation/docstrings. Volatility measures (such as ATR) must use proper smoothing methods if they serve as inputs for risk-management parameters (stop-losses, target levels, or position sizes).

### 8. Shadow Trade Direction-Awareness
*   **The Anti-Pattern**: Hardcoding long-only P&L calculations (`(price - entry) / entry * 100`) in counterfactual trackers (like shadow trades) because the current active configuration only tracks buys. If short signals are later enabled or recorded, their P&Ls become inverted, poisoning the ML calibration datasets with corrupted labels.
*   **The Rule**: Always calculate shadow/paper trade P&Ls using a direction-aware multiplier (e.g. `multiplier = 1 if LONG else -1`) to ensure schema columns and ML model retraining logic remain structurally consistent and future-proof.

### 9. System Outage vs. Market Risk Decoupling
*   **The Anti-Pattern**: Treating technical system failures (e.g., recommender scan timeouts or API outages) as standard market caution flags in decision arrays, or modifying final decision outcomes post-hoc. This leads to double-counting cautions and mismatching the reasoning states stored in the database or rendered in the UI (e.g., a "RED / STAND DOWN" verdict displaying a low caution flag count).
*   **The Rule**: Keep technical system health overrides completely decoupled from market risk evaluation decision tables. Check system-level failures first as a hard short-circuit/override. Do not append technical errors to standard caution/risk flags, ensuring that caution counts, actions, and labels are always atomically aligned and clean for UI rendering and calibration history.

### 11. Thread-Safe Database & Network Operations (SQLite Concurrency & Timezones)
*   **The Anti-Pattern**: Executing database write operations (like SQLite cache updates) inside parallel worker threads (e.g. `ThreadPoolExecutor`). This leads to concurrent write contention, causing `sqlite3.OperationalError: database is locked` errors. Similarly, making repetitive synchronous API requests inside nested loops (e.g., checking pairwise combinations) poisons execution time and gets the server rate-limited or blocked.
*   **The Rule**: Keep parallel thread pools read-only with respect to the database. Use threads exclusively for non-blocking network fetches (like `yfinance` history). Alignment, calculation, and database writes must be serialized and executed on the main thread. When pre-computing combination metrics, pre-fetch all required raw data once in parallel first, then perform pairwise operations in memory sequentially.
*   **The Rule (Timezones)**: Always use `datetime.utcnow()` instead of `datetime.now()` when checking expiration/staleness of database timestamps stored using SQLite's `datetime('now')` to avoid timezone offset discrepancies (e.g., IST vs UTC).

---

## Implemented Changes by Subsystem

### 1. Recommender Engine & Signal Generation
*   **Gap Strategy Alignment**: Fixed the direction logic for gap-down filled signals to map bullish recovery (weight `+1.5`, `"Gap Down (Filled)"`) in [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py) and [performance.py](file:///d:/Code/indian-trading-agent/backend/performance.py). Corrected gap-up open unfilled signals to emit short trades unconditionally on green candle days. Redefined gap fill detection to utilize intraday price extremes (`current_high`/`current_low`) instead of EOD closes in [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py), [performance.py](file:///d:/Code/indian-trading-agent/backend/performance.py), and [scanner.py](file:///d:/Code/indian-trading-agent/backend/scanner.py).
*   **Technical Indicator Accuracy**: Implemented Wilder's RSI calculation with exponential smoothing over the full historical closes series, fixing array slicing bugs in `_compute_rsi` and removing the simple-average 15-day lookup logic. Corrected [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py) `compute_atr` to use proper Wilder's exponential smoothing instead of a simple arithmetic mean, ensuring stop-losses are not set too tight during high-volatility regimes.
*   **Atomic Filter Workflows**: Unified recommender filter logic (Market Bias, Sector Concentration, Event Risks) in [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py). Moved post-filter evaluations into a shared atomic helper `_recompute_assessment_and_trade_plan()` to avoid stale target/stop-loss prices and out-of-sync conviction tags.
*   **Sector & Event Penalties**: Extended macro event calendars to check look-ahead periods (≤1 day for RBI rate policy, ≤2 days for FOMC). Implemented sector-aware event penalties in [calendar_data.py](file:///d:/Code/indian-trading-agent/backend/calendar_data.py), enforcing universal full penalties for market-wide events (such as the Union Budget) across all sectors by removing them from sector-specific sensitivity lists. Implemented sector concentration cap checks using trade-specific position sizes instead of hardcoded 10% cash assumptions in [concentration.py](file:///d:/Code/indian-trading-agent/backend/concentration.py).
*   **Error Management**: Replaced silent exception swallowing in `_analyze_stock()` with structured logging and surfaced skipped/failed tickers to the UI layer.
*   **Correlation-Aware Anti-Clustering Filter**: Implemented rolling pairwise correlation filtering on bullish signals to prevent redundant positions (high pairwise correlation) in [recommender.py](file:///d:/Code/indian-trading-agent/backend/recommender.py) and [concentration.py](file:///d:/Code/indian-trading-agent/backend/concentration.py). Standardized correlation cache retrieval to compare SQLite UTC timestamps with `datetime.utcnow()`. Reorganized the pre-computation cron tasks in [cron.py](file:///d:/Code/indian-trading-agent/backend/cron.py) to pre-fetch returns in parallel and run pairwise computations sequentially on the main thread.

### 2. Forecasting Model & Calibration
*   **L1-Regularized Logistic Regression**: Retired the legacy linear additive scoring logic. Created a 28-dimensional logistic model inside [signal_model.py](file:///d:/Code/indian-trading-agent/backend/signal_model.py) using an in-house Iterative Soft Thresholding Algorithm (ISTA) optimizer. Implemented 5-fold cross-validation with safety check promotion criteria ($AUC > 0.55$, $Brier < 0.20$).
*   **Three-Tier Honest Forecasting**: Replaced simple probability formulas with [HonestAssessmentEngine](file:///d:/Code/indian-trading-agent/backend/honest_assessment.py) separating data levels into four quality tiers (EXPLORATORY, EMERGING, EMPIRICAL, CALIBRATED), returning nulls gracefully for insufficient data and Wilson confidence intervals for empirical data.
*   **Position Sizing (Kelly Criterion)**: Replaced simplified $2p-1$ sizing with the full Kelly formula $(p \cdot b - q) / b$ computed from active trade plan risk/reward ratios. Implemented a 15% individual position cap, a 10% compounding portfolio drawdown override ceiling (sizing forced to 0%), and negative fraction overrides. Adjusted calculations for strategies with perfect wins (zero historical losses) by verifying wins and using safety defaults to prevent division errors.
*   **Data Integrity & Cache Priming**: Handled fingerprint mismatch pipeline bugs by isolating base strategy configurations from post-filter additions during database cache generation. Deduplicated paper and shadow trades on `(ticker, entry_date, fingerprint)` during cron cache rebuilds, training runs, and Kelly statistics evaluations to avoid double-counting.

### 3. Simulation & Backtesting Engine
*   **Market Calendar Advancement**: Replaced calendar-day increments with a custom `next_trading_day()` utility inside [simulation.py](file:///d:/Code/indian-trading-agent/backend/simulation.py) and [verdict_calibration.py](file:///d:/Code/indian-trading-agent/backend/verdict_calibration.py) to properly handle NSE market holidays and weekends, resolving Monday lock-in bias.
*   **Real-Time Simulation Mechanics**: Updated the backtest replay to process gap detection using intraday extremes (`current_low`/`current_high`) and use static `DEFAULT_WEIGHTS` (since weight overrides are retired). Resolved a calendar-date skip loop bug by advancing dates uniformly.
*   **Trade Expirations**: Modified paper/shadow trade auto-expirations to count trading days elapsed (`trading_days_elapsed` from [market_calendar.py](file:///d:/Code/indian-trading-agent/tradingagents/utils/market_calendar.py)) rather than calendar days to prevent early expirations.
*   **Drawdown Calculations**: Rewrote `get_portfolio_drawdown()` to compound equity based on actual position size percentages, incorporate mark-to-market unrealized P&L (updating quotes dynamically via `yfinance` if cache is older than 15 minutes), handle stop-loss hits from the database, sort same-day entry-exits correctly, and ignore NULL P&L records.
*   **Execution Safeguards**: Fixed parameter overwriting bug in `open_paper_trade()` to preserve user-specified stop-loss and risk/reward parameters, and ensured neutral backtest scores (-2.0 to 2.0) emit `"HOLD"` signals instead of silent nulls.

### 4. Database, Cron & Schema Migrations
*   **Startup Migrations**: Added self-healing database migrations to dynamically verify and add `position_size_pct`, `unrealized_pnl_pct`, `stop_loss_price`, and `risk_reward_ratio` columns to SQLite tables during startup `ensure_db()`.
*   **Performance Optimization**: Resolved N+1 query loops when fetching shadow or paper trades by batching signal performance cache queries using `IN` filters and implementing memory caching for identical configurations.
*   **Cron Daemon & Thread Safety**: Created a background cron daemon in [cron.py](file:///d:/Code/indian-trading-agent/backend/cron.py) to daily rebuild cache tables, weekly retrain calibration models, and check auto-stop-loss conditions. Implemented thread-safe `_RETRAIN_LOCK` around ML training passes and decoupled checks from price refresh loops.
*   **Fallback Resolution**: Primed signal performance caches during migrations, and added direct Python-based parsing for null fingerprints to fall back dynamically on JSON signals and entry regimes rather than failing back to exploratory defaults.
*   **Shadow Trade Direction-Aware P&L**: Updated `refresh_shadow_prices()` in `backend/shadow_trades.py` to fetch the `signal` direction from the database and apply a direction multiplier when calculating horizon P&Ls, preventing silent label inversion during signal model retraining.

### 5. UI, API & Daily Verdict Architecture
*   **API Deprecations**: Retired legacy manual weight overrides and weight-tuning API endpoints (`/apply`, `/reset`) by raising HTTP 400 Bad Request, and fully deprecated/removed overrides logic in recommender scoring and backtests (Option A).
*   **Fail-Safe Verdicts**: Restructured [daily_verdict.py](file:///d:/Code/indian-trading-agent/backend/daily_verdict.py) to force red stand-down verdicts when the underlying recommender engine throws exceptions, and bypassed stock-level filters during scan passes to prevent double-penalizing risk adjustments. Handled recommender failures as a clean decision logic override rather than a post-decision override to prevent caution count desynchronization and double-counting issues.
*   **Frontend Dashboard Pages**:
    *   **Simulation & Performance**: Rendered `"HOLD"` signals (colored gray, excluded from win rate calculations) on the simulation page, and added live market `<RegimeBadge />` updates and model coefficient tables on the Signal Performance page.
    *   **Risk & Position badges**: Rendered `HonestAssessmentBadge` and trade plans with entry, target, and stop-loss levels. Added a pre-trade checkbox on the dashboard to require risk acknowledgment before tracking paper trades.
    *   **Error banners**: Surfaced warning panels when yfinance limits/skipped tickers are returned in recommendations.




## Rejected Changes

- **Baseline Volatility Samples Wrong Historical Windows**: Assumed that the baseline volatility calculation in `backend/market_regime.py::classify_regime_for_date()` samples from random disjoint historical windows and proposed replacing `closes[:-i]` with `closes[-i-20:-i]`. This was rejected because the current code already computes rolling, overlapping 20-day volatilities spaced 5 days apart. Furthermore, the suggested fix slices an array of exactly 20 elements, which fails the internal length check in `_annualized_vol()` (which requires `window + 1` or 21 elements) and would cause it to always return `0.0`, breaking the `HIGH_VOL` regime detector. Even if corrected to 21 elements (`closes[-i-21:-i]`), the result is index-wise identical to the current code.
