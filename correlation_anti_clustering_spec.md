# Correlation-Aware Position Sizing (Anti-Clustering Filter)
## Implementation Specification for `vijayanajay/indian-trading-agent`

---

## 1. Overview

**Feature Name:** Correlation-Aware Position Sizing (Anti-Clustering Filter)
**Goal:** Prevent the recommender from suggesting multiple stocks that move together (high pairwise correlation), even if they pass the existing sector concentration check.
**Why:** A portfolio with RELIANCE + ONGC + BPCL + IOC + NTPC passes sector checks (all "Energy") but also a "diversified" portfolio with INFY + TCS + WIPRO + HCLTECH + TECHM is equally dangerous — both are effectively single-factor bets. The existing sector checker catches the first but misses the second if sectors are labeled differently (e.g., "IT" vs "Software"). Correlation is the ground truth.

**Effort Estimate:** ~80 lines across 4 files + 1 new table
**External Dependencies:** None (uses existing yfinance + numpy + pandas)

---

## 2. Files to Touch

| # | File | Action | Lines | Description |
|---|------|--------|-------|-------------|
| 1 | `backend/db.py` | Add `correlation_cache` table + helpers | ~25 | Schema setup and cache persistence |
| 2 | `backend/concentration.py` | Add aligned correlation computations | ~55 | Filter logic with date alignment and cache safety |
| 3 | `backend/recommender.py` | Wire `_apply_correlation_filter` | ~15 | Inject penalties into the active recommendation workflow |
| 4 | `backend/daily_verdict.py` | Update lightweight scan options | ~10 | Ensure scans bypass network fetches completely |
| 5 | `backend/routers/concentration.py` | Add API endpoints | ~15 | Expose correlation matrices and clustering diagnostics |

---

## 3. Detailed Implementation

---

### 3.1 `backend/db.py` — Schema & Cache

#### 3.1.1 Add to `ensure_db()`

Insert this table creation block inside the `conn.executescript()` call, after the `model_coefficients` table:

```sql
-- Correlation cache: 90-day rolling pairwise correlations between tickers
-- Refreshed daily by cron. Stale entries older than 7 days are ignored.
CREATE TABLE IF NOT EXISTS correlation_cache (
    ticker_a TEXT NOT NULL,
    ticker_b TEXT NOT NULL,
    correlation REAL NOT NULL,          -- Pearson correlation [-1, 1]
    lookback_days INTEGER DEFAULT 90,
    computed_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (ticker_a, ticker_b, lookback_days)
);
CREATE INDEX IF NOT EXISTS idx_corr_ticker_a ON correlation_cache(ticker_a);
CREATE INDEX IF NOT EXISTS idx_corr_computed_at ON correlation_cache(computed_at);
```

#### 3.1.2 Add Helper Functions (append to end of `db.py`)

```python
# ============================================================
# Correlation Cache
# ============================================================

def save_correlation(ticker_a: str, ticker_b: str, correlation: float, lookback_days: int = 90):
    """Save or update a pairwise correlation. Ensures canonical ordering (A < B)."""
    a, b = sorted([ticker_a.upper(), ticker_b.upper()])
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO correlation_cache
            (ticker_a, ticker_b, correlation, lookback_days, computed_at)
            VALUES (?, ?, ?, ?, datetime('now'))""",
            (a, b, correlation, lookback_days),
        )

def get_correlation(ticker_a: str, ticker_b: str, lookback_days: int = 90):
    """Get cached correlation. Returns None if not cached or stale (>7 days)."""
    a, b = sorted([ticker_a.upper(), ticker_b.upper()])
    with get_db() as conn:
        row = conn.execute(
            """SELECT correlation, computed_at FROM correlation_cache
            WHERE ticker_a = ? AND ticker_b = ? AND lookback_days = ?""",
            (a, b, lookback_days),
        ).fetchone()
        if not row:
            return None
        try:
            computed = datetime.fromisoformat(row["computed_at"])
            # Compare against UTC to match SQLite's datetime('now')
            if (datetime.utcnow() - computed).days > 7:
                return None  # Stale
        except Exception:
            return None
        return row["correlation"]

def get_correlations_for_ticker(ticker: str, lookback_days: int = 90):
    """Get all cached correlations for a ticker. Returns {other_ticker: correlation}."""
    ticker = ticker.upper()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ticker_a, ticker_b, correlation, computed_at FROM correlation_cache
            WHERE (ticker_a = ? OR ticker_b = ?) AND lookback_days = ?""",
            (ticker, ticker, lookback_days),
        ).fetchall()
        result = {}
        for r in rows:
            try:
                computed = datetime.fromisoformat(r["computed_at"])
                # Compare against UTC to match SQLite's datetime('now')
                if (datetime.utcnow() - computed).days > 7:
                    continue
            except Exception:
                continue
            other = r["ticker_b"] if r["ticker_a"] == ticker else r["ticker_a"]
            result[other] = r["correlation"]
        return result

def prune_stale_correlations(max_age_days: int = 7):
    """Delete correlation entries older than N days. Parameterized safely for SQLite."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM correlation_cache WHERE computed_at < datetime('now', ?)",
            (f"-{max_age_days} days",),
        )
```

---

### 3.2 `backend/concentration.py` — Correlation Engine

#### 3.2.1 Add Imports

At the top of `backend/concentration.py`, add:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tradingagents.utils.ticker import normalize_ticker
from backend.db import get_correlation, save_correlation, get_db
```

#### 3.2.2 Add Core Correlation Function

Append to `backend/concentration.py`:

```python
# ============================================================
# CORRELATION-AWARE ANTI-CLUSTERING
# ============================================================

CORRELATION_LOOKBACK_DAYS = 90   # ~3 months of trading data
HIGH_CORRELATION_THRESHOLD = 0.70  # Flag if avg correlation > 70%
MAX_CORRELATED_POSITIONS = 2       # Allow max 2 positions with avg corr > 0.70

def _fetch_returns(ticker: str, days: int = CORRELATION_LOOKBACK_DAYS) -> pd.Series | None:
    """Fetch daily returns for a ticker over N days. Returns pd.Series with DatetimeIndex or None on failure."""
    try:
        symbol = normalize_ticker(ticker)
        t = yf.Ticker(symbol)
        # Add buffer for weekends/holidays
        hist = t.history(period=f"{int(days * 1.5)}d")
        if hist.empty or len(hist) < days // 2:
            return None
        hist = hist.dropna(subset=["Close"])
        returns = hist["Close"].pct_change().dropna()
        if len(returns) < 20:
            return None
        # Ensure timezone-naive DatetimeIndex for perfect alignment
        if returns.index.tz is not None:
            returns.index = returns.index.tz_convert(None)
        return returns.tail(days)
    except Exception:
        return None

def compute_pairwise_correlation(
    ticker_a: str,
    ticker_b: str,
    ret_a: pd.Series = None,
    ret_b: pd.Series = None,
    fetch_if_missing: bool = True,
) -> float | None:
    """Compute Pearson correlation between two tickers using date-aligned Pandas series. Uses cache if fresh."""
    # Check cache first
    cached = get_correlation(ticker_a, ticker_b, CORRELATION_LOOKBACK_DAYS)
    if cached is not None:
        return cached

    if not fetch_if_missing:
        return None

    # Fetch returns if not provided pre-aligned
    if ret_a is None:
        ret_a = _fetch_returns(ticker_a)
    if ret_b is None:
        ret_b = _fetch_returns(ticker_b)

    if ret_a is None or ret_b is None:
        return None

    # Crucial: Align indices (dates) using dictionary keys to prevent column naming overlap warnings
    df = pd.concat({"a": ret_a, "b": ret_b}, axis=1, join="inner")
    if len(df) < 20:
        return None

    # Compute date-aligned Pearson correlation
    corr = float(df["a"].corr(df["b"]))
    if np.isnan(corr):
        return None

    # Cache result
    save_correlation(ticker_a, ticker_b, corr, CORRELATION_LOOKBACK_DAYS)
    return corr

def get_avg_correlation_with_portfolio(ticker: str, open_tickers: list[str], fetch_if_missing: bool = True):
    """Compute average correlation of `ticker` against all open positions.
    
    Checks cache first. If missing and fetch_if_missing is True, fetches candidate returns
    once and open position returns in parallel using a ThreadPoolExecutor.
    """
    if not open_tickers:
        return {
            "avg_correlation": None,
            "pairwise": {},
            "max_correlation": None,
            "max_correlation_ticker": None,
            "n_computed": 0,
            "n_total": 0,
        }

    pairwise = {}
    missing_tickers = []

    # Satisfy cache lookups first
    for ot in open_tickers:
        cached = get_correlation(ticker, ot, CORRELATION_LOOKBACK_DAYS)
        if cached is not None:
            pairwise[ot] = cached
        else:
            missing_tickers.append(ot)

    # Perform thread-safe parallel fetches only for missing entries, fetching candidate ONCE.
    # To prevent SQLite write lock contention, threads only fetch data from yfinance;
    # alignment and DB cache saves are performed sequentially on the main thread.
    if missing_tickers and fetch_if_missing:
        ret_candidate = _fetch_returns(ticker)
        if ret_candidate is not None:
            returns_map = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_fetch_returns, ot): ot for ot in missing_tickers}
                for f in as_completed(futures):
                    ot = futures[f]
                    try:
                        ret_ot = f.result()
                        if ret_ot is not None:
                            returns_map[ot] = ret_ot
                    except Exception:
                        pass

            # Compute and save correlations on the main thread sequentially
            for ot, ret_ot in returns_map.items():
                corr = compute_pairwise_correlation(ticker, ot, ret_candidate, ret_ot, fetch_if_missing=True)
                if corr is not None:
                    pairwise[ot] = corr

    if not pairwise:
        return {
            "avg_correlation": None,
            "pairwise": {},
            "max_correlation": None,
            "max_correlation_ticker": None,
            "n_computed": 0,
            "n_total": len(open_tickers),
        }

    correlations = list(pairwise.values())
    avg_corr = float(np.mean(correlations))
    max_corr = max(correlations)
    max_ticker = max(pairwise, key=pairwise.get)

    return {
        "avg_correlation": round(avg_corr, 3),
        "pairwise": {k: round(v, 3) for k, v in pairwise.items()},
        "max_correlation": round(max_corr, 3),
        "max_correlation_ticker": max_ticker,
        "n_computed": len(pairwise),
        "n_total": len(open_tickers),
    }

def check_correlation_clustering(
    ticker: str,
    total_capital: float = 500000,
    threshold: float = HIGH_CORRELATION_THRESHOLD,
    fetch_if_missing: bool = True,
):
    """Check if adding `ticker` would create a high-correlation cluster."""
    # Get open positions (same source as sector concentration)
    open_positions = get_open_positions()
    open_tickers = [p["ticker"] for p in open_positions if p.get("ticker")]

    # Exclude self if already in portfolio (shouldn't happen, but safety)
    open_tickers = [t for t in open_tickers if t.upper() != ticker.upper()]

    if not open_tickers:
        return {
            "ticker": ticker,
            "would_cluster": False,
            "avg_correlation": None,
            "max_correlation": None,
            "max_correlation_ticker": None,
            "cluster_tickers": [],
            "warnings": [],
            "score_adjustment": 0.0,
            "n_open_positions": 0,
            "n_computed": 0,
        }

    corr_data = get_avg_correlation_with_portfolio(ticker, open_tickers, fetch_if_missing=fetch_if_missing)
    avg_corr = corr_data["avg_correlation"]
    max_corr = corr_data["max_correlation"]
    max_ticker = corr_data["max_correlation_ticker"]

    warnings = []
    score_adj = 0.0
    would_cluster = False
    cluster_tickers = []

    if avg_corr is not None:
        # Count how many existing positions have correlation > threshold
        high_corr_partners = [
            t for t, c in corr_data["pairwise"].items()
            if abs(c) > threshold
        ]

        if len(high_corr_partners) >= MAX_CORRELATED_POSITIONS:
            would_cluster = True
            cluster_tickers = high_corr_partners[:MAX_CORRELATED_POSITIONS]
            warnings.append(
                f"High correlation cluster: {ticker} correlates "
                f"{corr_data['pairwise'][cluster_tickers[0]]:.0%} with "
                f"{cluster_tickers[0]}"
                + (f" and {corr_data['pairwise'][cluster_tickers[1]]:.0%} with {cluster_tickers[1]}"
                   if len(cluster_tickers) > 1 else "")
                + " — you're making one bet, not multiple."
            )
            score_adj -= 1.5
        elif avg_corr > threshold:
            would_cluster = True
            cluster_tickers = [max_ticker] if max_ticker else []
            warnings.append(
                f"{ticker} averages {avg_corr:.0%} correlation with your open positions "
                f"({max_corr:.0%} with {max_ticker}) — adds redundant risk, not diversification."
            )
            score_adj -= 1.0
        elif avg_corr > threshold * 0.8:
            # Soft warning at 80% of threshold
            warnings.append(
                f"{ticker} is approaching correlation limit: {avg_corr:.0%} avg "
                f"(threshold {threshold:.0%}). Consider a different sector."
            )
            score_adj -= 0.5

    return {
        "ticker": ticker,
        "would_cluster": would_cluster,
        "avg_correlation": avg_corr,
        "max_correlation": max_corr,
        "max_correlation_ticker": max_ticker,
        "cluster_tickers": cluster_tickers,
        "warnings": warnings,
        "score_adjustment": round(score_adj, 2),
        "n_open_positions": len(open_tickers),
        "n_computed": corr_data["n_computed"],
    }
```

#### 3.2.3 Add to `get_concentration_summary()`

In `get_concentration_summary()`, add a `fetch_if_missing` parameter and correlation summary:

```python
def get_concentration_summary(total_capital: float = 500000, fetch_if_missing: bool = True) -> dict:
    """High-level summary for dashboard display, modified to support optional correlation fetches."""
    allocation = get_sector_allocation(total_capital)

    # Top sectors by exposure
    sectors_sorted = sorted(
        [(name, data) for name, data in allocation["by_sector"].items()],
        key=lambda x: -x[1]["percent"],
    )

    top_sector = sectors_sorted[0] if sectors_sorted else None

    risk_level = "LOW"
    risk_reason = "Portfolio well diversified"

    if allocation["concentrated_sectors"]:
        risk_level = "HIGH"
        risk_reason = f"Over-concentrated in {', '.join(allocation['concentrated_sectors'])}"
    elif top_sector and top_sector[1]["percent"] > DEFAULT_MAX_PERCENT_PER_SECTOR * 0.8:
        risk_level = "MEDIUM"
        risk_reason = f"{top_sector[0]} approaching limit ({top_sector[1]['percent']:.1f}%)"
    elif allocation["total_positions"] == 0:
        risk_level = "NONE"
        risk_reason = "No open positions"

    # --- Correlation clustering summary ---
    # Check if top sector positions are actually a correlation cluster
    correlation_risk = "LOW"
    correlation_reason = "Positions are sufficiently diversified"
    
    if top_sector and top_sector[1]["count"] >= 2:
        # Sample up to 3 tickers from top sector for correlation check
        sample_tickers = [p["ticker"] for p in top_sector[1].get("positions", [])[:3]]
        if len(sample_tickers) >= 2:
            pair_corrs = []
            for i in range(len(sample_tickers)):
                for j in range(i + 1, len(sample_tickers)):
                    c = compute_pairwise_correlation(
                        sample_tickers[i],
                        sample_tickers[j],
                        fetch_if_missing=fetch_if_missing,
                    )
                    if c is not None:
                        pair_corrs.append(abs(c))
            
            if pair_corrs:
                mean_corr = np.mean(pair_corrs)
                if mean_corr > HIGH_CORRELATION_THRESHOLD:
                    correlation_risk = "HIGH"
                    correlation_reason = (
                        f"Your {top_sector[0]} positions move {mean_corr:.0%} together — "
                        "this is one concentrated bet, not diversification."
                    )
                elif mean_corr > HIGH_CORRELATION_THRESHOLD * 0.8:
                    correlation_risk = "MEDIUM"
                    correlation_reason = (
                        f"{top_sector[0]} positions are {mean_corr:.0%} correlated — "
                        "approaching cluster risk."
                    )

    return {
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "total_positions": allocation["total_positions"],
        "total_allocated_pct": allocation["total_allocated_pct"],
        "top_sector": {
            "name": top_sector[0],
            "count": top_sector[1]["count"],
            "percent": top_sector[1]["percent"],
        } if top_sector else None,
        "by_sector": allocation["by_sector"],
        "concentrated_sectors": allocation["concentrated_sectors"],
        "limits": allocation["limits"],
        "correlation_risk": correlation_risk,
        "correlation_reason": correlation_reason,
    }
```

---

### 3.3 `backend/recommender.py` — Wire Into Recommendation Flow

#### 3.3.1 Add Filter Application Function

Append near `_apply_concentration_filter()`:

```python
def _apply_correlation_filter(result: dict, correlation_check: dict) -> dict:
    """Apply correlation clustering penalty if the stock moves too similarly to existing positions."""
    if not correlation_check:
        return result

    adj = correlation_check.get("score_adjustment", 0)
    if adj == 0:
        return result

    new_score = round(result["score"] + adj, 2)
    warnings = correlation_check.get("warnings", [])

    corr_signal = {
        "type": "Correlation Clustering",
        "direction": "BEARISH",
        "value": "; ".join(warnings) if warnings else "High correlation with existing positions",
        "weight": adj,
        "metadata": {
            "avg_correlation": correlation_check.get("avg_correlation"),
            "max_correlation": correlation_check.get("max_correlation"),
            "max_correlation_ticker": correlation_check.get("max_correlation_ticker"),
            "cluster_tickers": correlation_check.get("cluster_tickers", []),
        },
    }
    result.setdefault("filter_adjustments", []).append(corr_signal)
    result["correlation_warning"] = "; ".join(warnings) if warnings else None
    result["correlation_breach"] = correlation_check.get("would_cluster", False)

    # Re-compute honest assessment, direction, and trade plan
    _recompute_assessment_and_trade_plan(result, new_score)

    return _recompute_confidence_and_counts(result, include_filters=True)
```

#### 3.3.2 Wire Into `recommend()`

In `recommend()`, after the concentration filter block (around line 720), add:

```python
                # Apply correlation clustering check (only on bullish signals)
                if apply_concentration_check and result.get("direction") in ("STRONG BUY", "BUY"):
                    try:
                        from backend.concentration import check_correlation_clustering
                        corr_check = check_correlation_clustering(
                            result["ticker"],
                            total_capital=total_capital,
                            fetch_if_missing=apply_correlation_check, # Propagate network fetch status
                        )
                        if corr_check:
                            result = _apply_correlation_filter(result, corr_check)
                    except Exception as e:
                        print(f"[Recommender] Correlation check failed: {e}", flush=True)
```

Also update the function signature of `recommend()` to include:

```python
def recommend(
    universe: str = "nifty100",
    min_signals: int = 2,
    apply_market_bias: bool = True,
    apply_event_filter: bool = True,
    apply_concentration_check: bool = True,
    apply_correlation_check: bool = True,   # <-- NEW
    total_capital: float = 500000,
) -> dict:
```

---

### 3.4 `backend/daily_verdict.py` — Bypassing Network Fetches

In `daily_verdict.py`, the lightweight scan is structured to execute without network overhead. The recommendation runs must skip correlation fetching, and the concentration calls must restrict themselves to the SQLite cache only.

```python
    # === 3. Sector Concentration ===
    try:
        from backend.concentration import get_concentration_summary
        # Bypasses yfinance network correlation fetches during the synchronous daily verdict check
        conc = get_concentration_summary(fetch_if_missing=False)
        filter_results["concentration"] = conc

        if conc["risk_level"] == "HIGH":
            caution_flags.append(f"Portfolio over-concentrated in {', '.join(conc.get('concentrated_sectors', []))}")
        elif conc["risk_level"] == "MEDIUM":
            caution_flags.append("Portfolio approaching sector limit")
    except Exception:
        filter_results["concentration"] = None

    # === 4. Quick scan: how many HIGH-conviction setups exist? ===
    ...
        recs = recommend(
            universe="nifty50",
            min_signals=2,
            apply_market_bias=False,
            apply_event_filter=False,
            apply_concentration_check=False,
            apply_correlation_check=False,   # <-- Skip network fetches for correlation
        )
```

---

### 3.5 `backend/routers/concentration.py` — API Endpoint

If the file doesn't exist, create it. Add:

```python
from fastapi import APIRouter
from backend.concentration import check_correlation_clustering, get_concentration_summary

router = APIRouter(prefix="/api/concentration", tags=["concentration"])

@router.get("/correlation/{ticker}")
def get_correlation_check(ticker: str, total_capital: float = 500000, fetch_if_missing: bool = True):
    """Check correlation clustering risk for a ticker against open positions."""
    return check_correlation_clustering(ticker, total_capital=total_capital, fetch_if_missing=fetch_if_missing)

@router.get("/summary")
def concentration_summary(total_capital: float = 500000, fetch_if_missing: bool = True):
    """Get full concentration + correlation summary."""
    return get_concentration_summary(total_capital=total_capital, fetch_if_missing=fetch_if_missing)
```

Then wire into `backend/app.py`:

```python
from backend.routers import concentration as concentration_router
# ...
app.include_router(concentration_router.router)
```

---

## 4. Frontend Integration (Optional but Recommended)

### 4.1 Dashboard Card Update

In the frontend Dashboard's concentration tracker card, add:

```tsx
{concentration.correlation_risk !== "LOW" && (
  <div className={`alert alert-${concentration.correlation_risk === "HIGH" ? "danger" : "warning"}`}>
    <strong>⚠️ Correlation Risk:</strong> {concentration.correlation_reason}
  </div>
)}
```

### 4.2 Top Picks Filter Badge

When rendering each pick in Top Picks, if `correlation_breach` is true:

```tsx
{pick.correlation_breach && (
  <span className="badge bg-warning text-dark" title={pick.correlation_warning}>
    🔗 Cluster Risk
  </span>
)}
```

---

## 5. Cron Job for Cache Warm-Up

In `backend/cron.py`, add to `_cron_loop()` inside the daily block:

```python
            # 3. Pre-compute correlations for NIFTY 50 pairs (expensive, do once daily)
            try:
                from backend.concentration import compute_pairwise_correlation, _fetch_returns
                from backend.scanner import NIFTY_100
                import itertools
                from concurrent.futures import ThreadPoolExecutor, as_completed
                # Only compute for top 50 to keep it fast (~1225 pairs)
                top_50 = NIFTY_100[:50]
                
                # Fetch returns for all 50 tickers once in parallel to avoid rate limits
                logger.info("Pre-fetching returns for NIFTY 50 tickers in parallel...")
                returns_cache = {}
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(_fetch_returns, ticker): ticker for ticker in top_50}
                    for f in as_completed(futures):
                        ticker = futures[f]
                        try:
                            ret = f.result()
                            if ret is not None:
                                returns_cache[ticker] = ret
                        except Exception as e:
                            logger.warning(f"Failed to fetch returns for {ticker}: {e}")
                
                # Compute all pairwise combinations using pre-fetched returns sequentially on main thread
                computed = 0
                for a, b in itertools.combinations(top_50, 2):
                    ret_a = returns_cache.get(a)
                    ret_b = returns_cache.get(b)
                    if ret_a is not None and ret_b is not None:
                        c = compute_pairwise_correlation(a, b, ret_a, ret_b, fetch_if_missing=False)
                        if c is not None:
                            computed += 1
                logger.info(f"Pre-computed {computed} correlations for NIFTY 50.")
            except Exception as e:
                logger.error(f"Correlation pre-computation failed: {e}")
```

---

## 6. Testing Checklist

| Test | Expected Result |
|------|-----------------|
| Open 2 IT positions (INFY, TCS). Recommend WIPRO. | Score penalty -1.5, "High correlation cluster" warning, direction may flip to NEUTRAL. |
| Open 1 Energy (RELIANCE). Recommend ONGC. | Soft warning if avg corr > 0.56, no penalty if < 0.70. |
| Empty portfolio. Recommend any stock. | No penalty, no warnings. |
| Cache hit: call `check_correlation_clustering("INFY", ...)` twice. | Second call uses cache, no yfinance call. |
| Stale cache (>7 days). | Re-fetches from yfinance, updates cache. |
| yfinance timeout for one ticker. | Gracefully skips that pair, computes avg from available data. |
| Daily verdict lightweight scan. | `fetch_if_missing=False`, `apply_correlation_check=False`, 0ms network overhead. |

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pandas Date Alignment** | Using `pd.concat` with `join="inner"` on DatetimeIndex guarantees comparison of identical trading days. Prevents offset lag math poisoning. Uses dictionary keys `{"a": ret_a, "b": ret_b}` to prevent column name collisions. |
| **Timezone-Naive Normalization** | Explicitly converting returns datetime index to timezone-naive (`tz_convert(None)`) guarantees clean date alignment and prevents Pandas index matching crashes. |
| **Thread-Safe SQLite Decoupling** | Decoupling yfinance network fetching (`_fetch_returns`) from SQLite writes ensures that parallel execution inside `ThreadPoolExecutor` does not create database write contention or `database is locked` errors. All DB writes are serialized sequentially on the main thread. |
| **Timezone-Safe Cache Comparison** | Cache age calculation compares SQLite UTC timestamps (`datetime('now')`) with `datetime.utcnow()` to prevent local time offset mismatches (e.g. 5.5 hours in IST). |
| **Cron Caching Optimization** | Pre-fetching returns for NIFTY 50 once in parallel instead of fetching them on every combination check. This reduces network requests from 2,450 to 50, avoiding yfinance API rate-limiting and speeding up execution by ~100x. |
| **Unconditional Filter Scoring** | Applying the correlation filter unconditionally (instead of wrapping it in a `would_cluster` conditional check) ensures that soft warnings (with `-0.5` adjustment) are not silently dropped. |
| **Candidate Single-Fetch Workflow** | Pre-fetching target candidate returns *before* querying open portfolio positions in parallel prevents redundant external API invocations. |
| **Separated Online/Cache Scans** | Adding `fetch_if_missing` controls ensures synchronous pathways (Daily Verdict) do not stall on external dependencies. |
| **90-day lookback** | Balances responsiveness (catches recent regime shifts) with stability (not noise). 90 trading days ≈ 4 months. |
| **0.70 threshold** | Industry standard for "high correlation." Stocks with >0.70 correlation provide minimal diversification benefit. |
| **Max 2 correlated positions** | Allows a "pair trade" or "theme bet" (2 stocks) but blocks clusters of 3+. |
| **Cache 7-day TTL** | Correlations are slow-moving. Daily warm-up keeps cache fresh without hammering yfinance. |

---

## 8. Performance Impact

| Scenario | Latency |
|----------|---------|
| Cold cache, 10 open positions | ~3s (1 candidate fetch + 10 portfolio fetches in parallel) |
| Warm cache, 10 open positions | ~10ms (SQLite lookups only) |
| Daily verdict (lightweight scan) | 0ms (skipped or cache-only) |
| Cron pre-computation (daily) | ~60s background, no user impact |

---

## 9. Future Extensions (Out of Scope for V1)

1. **Factor correlation** (beta to Nifty, USD/INR, crude) instead of just pairwise stock correlation.
2. **Dynamic threshold** based on market regime (higher threshold in HIGH_VOL since everything correlates).
3. **Correlation decay** — weight recent months more heavily in the 90-day window.
4. **Sector-ETF correlation** — check correlation to sector ETFs for a cleaner factor read.

---

## 10. Rollback Plan

If the feature causes issues:

1. Set `apply_correlation_check=False` in `recommend()` default args.
2. Or: set `HIGH_CORRELATION_THRESHOLD = 1.0` (effectively disables).
3. Or: delete `correlation_cache` table and revert the file changes.

No database migrations needed beyond the single table creation (idempotent `CREATE TABLE IF NOT EXISTS`).

---

## 11. Implementation Verification Checklist (Added on 2026-06-14 by Kailash Nadh)

### Backend Services
- [x] **Schema & Cache (SQLite)**: Table `correlation_cache` correctly defined and initialized in `backend/db.py`.
- [x] **Database Helpers**: `save_correlation`, `get_correlation`, `get_correlations_for_ticker`, and `prune_stale_correlations` appended to `backend/db.py`.
- [x] **Correlation Engine**: Daily returns fetching, Pandas Datetime date alignment, and parallel candidate fetches via ThreadPoolExecutor implemented in `backend/concentration.py`.
- [x] **Recommender Integration**: Score penalty (-0.5 to -1.5) applied for highly correlated candidates in `backend/recommender.py` and confidence/assessment re-computed honesty.
- [x] **Fast/Offline Pathways**: `daily_verdict.py` and daily verdict scans bypass network fetches correctly (`fetch_if_missing=False` or `apply_correlation_check=False`).
- [x] **API Endpoints**: `backend/routers/concentration.py` exposes `/api/concentration/correlation/{ticker}` and `/api/concentration/summary`, and router is registered in `backend/app.py`.
- [x] **Cron Warm-Up**: NIFTY 50 pre-computation task implemented and scheduled in `backend/cron.py` to prevent real-time latency hits on recommendation scans.
- [x] **Test Coverage**: Basic test coverage for caching, returns alignment, clustering, and score adjustments added to `tests/backend/test_concentration.py`.

### Frontend Integration (Optional but Recommended)
- [x] **Dashboard Widget Update**: `ConcentrationWidget.tsx` displays `correlation_risk` warnings and `correlation_reason` alerts when risk level is not LOW or NONE.
- [x] **Top Picks & Recommendations Badges**: `TodayPicks.tsx` and `recommendations/page.tsx` render the `Cluster Risk` badge when `correlation_breach` is true.

### Summary Verdict
**Status:** Both backend services and frontend user interface elements are fully complete, robust, and correctly implemented.
