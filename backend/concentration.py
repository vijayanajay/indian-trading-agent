"""Sector Concentration Checker.

Prevents the AI from over-exposing to a single sector. Critical for autonomous trading
because 5 BUY signals in one morning could all be in the same sector — making them
essentially one trade with 5x the risk.

Sources of "open positions" we track:
1. Paper trades (status='active') from simulation table
2. Open analysis trades (pnl_status='open') from analysis_history
3. Watchlist (optional — for "what if" scenarios)

Outputs:
- Current allocation %  per sector
- Warnings when proposed new trades would push concentration too high
- Score adjustment for the recommendation engine to penalize concentrated bets
"""

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional
from backend.cyclical import SECTOR_MAP
from backend.db import (
    list_paper_trades,
    get_analysis_history,
    get_correlation,
    save_correlation,
    get_db,
)
from tradingagents.utils.ticker import normalize_ticker


# Build reverse map: ticker -> sector for quick lookup
TICKER_TO_SECTOR: dict[str, str] = {}
for sector_name, tickers in SECTOR_MAP.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker.upper()] = sector_name

# Default limits (configurable via settings table later)
DEFAULT_MAX_POSITIONS_PER_SECTOR = 3
DEFAULT_MAX_PERCENT_PER_SECTOR = 30.0  # % of total portfolio capital


def get_sector_for_ticker(ticker: str) -> str:
    """Get the sector for a ticker. Returns 'Other' if unknown."""
    return TICKER_TO_SECTOR.get(ticker.upper(), "Other")


def get_open_positions() -> list[dict]:
    """Get all open positions across paper trades + open analysis trades.
    
    Deduplicates positions by ticker (case-insensitive) to prevent double-counting.
    If a ticker has both an active paper trade and an open analysis trade, they 
    are merged into a single position entry keeping the maximum of their 
    effective position sizes.
    """
    merged: dict[str, dict] = {}

    # Paper trades that are active
    for pt in list_paper_trades(status="active"):
        ticker = pt.get("ticker", "").upper()
        if not ticker:
            continue
        
        pos_dict = {
            "id": f"paper-{pt.get('id')}",
            "ticker": ticker,
            "sector": get_sector_for_ticker(ticker),
            "direction": pt.get("direction", "LONG"),
            "entry_price": pt.get("entry_price"),
            "entry_date": pt.get("entry_date"),
            "source": "paper_trade",
            "strategy": pt.get("strategy"),
            "position_size_pct": pt.get("position_size_pct"),
        }
        
        if ticker in merged:
            existing = merged[ticker]
            size_existing = existing.get("position_size_pct")
            val_existing = size_existing if size_existing is not None else 10.0
            
            size_new = pos_dict.get("position_size_pct")
            val_new = size_new if size_new is not None else 10.0
            
            existing["position_size_pct"] = max(val_existing, val_new)
            existing["source"] = "merged"
        else:
            merged[ticker] = pos_dict

    # Real analyses with open status
    for ah in get_analysis_history(limit=200):
        if ah.get("pnl_status") == "open":
            ticker = ah.get("ticker", "").upper()
            if not ticker:
                continue
            
            real_pos = {
                "id": f"real-{ah.get('task_id')}",
                "ticker": ticker,
                "sector": get_sector_for_ticker(ticker),
                "direction": "LONG",  # default; signal-based inference could refine
                "entry_price": ah.get("entry_price"),
                "entry_date": ah.get("trade_date"),
                "source": "real_trade",
                "signal": ah.get("signal"),
            }
            
            if ticker in merged:
                existing = merged[ticker]
                size_existing = existing.get("position_size_pct")
                val_existing = size_existing if size_existing is not None else 10.0
                
                val_real = 10.0
                
                existing["position_size_pct"] = max(val_existing, val_real)
                existing["source"] = "merged"
                if "signal" in real_pos and "signal" not in existing:
                    existing["signal"] = real_pos["signal"]
            else:
                merged[ticker] = real_pos

    return list(merged.values())


def get_sector_allocation(total_capital: float = 500000) -> dict:
    """Compute current sector allocation across all open positions.

    Returns:
        {
            "total_positions": int,
            "by_sector": {sector: {count, percent, positions: [...]}},
            "concentrated_sectors": list of sectors over the limit,
            "total_allocated_pct": float
        }
    """
    positions = get_open_positions()

    by_sector: dict[str, list] = defaultdict(list)
    total_value = 0.0

    for pos in positions:
        sector = pos.get("sector", "Other")
        # Position value approximation — use actual position_size_pct if available, fallback to 10%
        actual_size_pct = pos.get("position_size_pct")
        if actual_size_pct is not None:
            position_value = total_capital * (actual_size_pct / 100.0)
        else:
            position_value = total_capital / 10  # fallback
        pos["position_value"] = position_value
        by_sector[sector].append(pos)
        total_value += position_value

    # Compute percentages
    sectors_summary = {}
    concentrated = []

    for sector, pos_list in by_sector.items():
        sector_value = sum(p.get("position_value", 0) for p in pos_list)
        percent = (sector_value / total_capital * 100) if total_capital > 0 else 0
        sectors_summary[sector] = {
            "count": len(pos_list),
            "percent": round(percent, 1),
            "value": round(sector_value, 0),
            "positions": pos_list,
        }
        if percent > DEFAULT_MAX_PERCENT_PER_SECTOR or len(pos_list) > DEFAULT_MAX_POSITIONS_PER_SECTOR:
            concentrated.append(sector)

    return {
        "total_positions": len(positions),
        "total_capital": total_capital,
        "total_allocated": round(total_value, 0),
        "total_allocated_pct": round((total_value / total_capital * 100) if total_capital > 0 else 0, 1),
        "by_sector": sectors_summary,
        "concentrated_sectors": concentrated,
        "limits": {
            "max_positions_per_sector": DEFAULT_MAX_POSITIONS_PER_SECTOR,
            "max_percent_per_sector": DEFAULT_MAX_PERCENT_PER_SECTOR,
        },
    }


def check_new_trade_concentration(
    ticker: str,
    proposed_position_value: Optional[float] = None,
    total_capital: float = 500000,
) -> dict:
    """Check if adding a new trade would breach concentration limits.

    Args:
        ticker: ticker being considered
        proposed_position_value: how much capital this trade would use
        total_capital: total portfolio capital

    Returns:
        {
            "ticker": str,
            "sector": str,
            "current_sector_count": int,
            "current_sector_percent": float,
            "would_breach": bool,
            "warnings": list[str],
            "score_adjustment": float (negative if concentrated),
        }
    """
    sector = get_sector_for_ticker(ticker)
    if not proposed_position_value:
        proposed_position_value = total_capital / 10  # default 10% per position

    allocation = get_sector_allocation(total_capital)
    sector_data = allocation["by_sector"].get(sector, {"count": 0, "percent": 0, "positions": []})

    new_count = sector_data["count"] + 1
    new_percent = sector_data["percent"] + (proposed_position_value / total_capital * 100)

    warnings = []
    score_adj = 0.0
    would_breach = False

    if new_count > DEFAULT_MAX_POSITIONS_PER_SECTOR:
        warnings.append(
            f"Would exceed max positions per sector ({new_count} > {DEFAULT_MAX_POSITIONS_PER_SECTOR}) in {sector}"
        )
        score_adj -= 1.5
        would_breach = True

    if new_percent > DEFAULT_MAX_PERCENT_PER_SECTOR:
        warnings.append(
            f"Would push {sector} to {new_percent:.1f}% of portfolio (max {DEFAULT_MAX_PERCENT_PER_SECTOR}%)"
        )
        score_adj -= 1.5
        would_breach = True

    # Soft warning at 80% of limit
    if not would_breach:
        if new_count >= DEFAULT_MAX_POSITIONS_PER_SECTOR:
            warnings.append(f"At max positions for {sector} ({new_count}/{DEFAULT_MAX_POSITIONS_PER_SECTOR})")
            score_adj -= 0.5
        elif new_percent >= DEFAULT_MAX_PERCENT_PER_SECTOR * 0.8:
            warnings.append(f"Approaching sector limit: {sector} would be {new_percent:.1f}%")
            score_adj -= 0.5

    return {
        "ticker": ticker,
        "sector": sector,
        "current_sector_count": sector_data["count"],
        "current_sector_percent": sector_data["percent"],
        "new_sector_count": new_count,
        "new_sector_percent": round(new_percent, 1),
        "would_breach": would_breach,
        "warnings": warnings,
        "score_adjustment": round(score_adj, 2),
        "existing_in_sector": [p["ticker"] for p in sector_data.get("positions", [])],
    }


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

