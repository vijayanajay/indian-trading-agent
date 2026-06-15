"""Strategy endpoints — Support/Resistance, Pivot Points, and strategy-based signals."""

from fastapi import APIRouter, Query
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from tradingagents.utils.ticker import normalize_ticker
from backend.cyclical import (
    analyze_monthly_seasonality,
    analyze_day_of_week,
    analyze_sector_rotation,
    backtest_seasonal_strategy,
)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


def _calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """Calculate classic pivot points: PP, S1-S3, R1-R3."""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return {
        "pivot": round(pp, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
    }


def _find_support_resistance(highs: list, lows: list, closes: list, n_levels: int = 3) -> dict:
    """Find key support/resistance levels from price history using local extremes."""
    prices = np.array(closes)
    all_highs = np.array(highs)
    all_lows = np.array(lows)
    current_price = prices[-1]

    # Find local peaks (resistance) and troughs (support)
    window = max(3, len(prices) // 10)

    resistance_levels = []
    support_levels = []

    for i in range(window, len(all_highs) - window):
        # Local high — potential resistance
        if all_highs[i] == max(all_highs[i - window:i + window + 1]):
            resistance_levels.append(float(all_highs[i]))
        # Local low — potential support
        if all_lows[i] == min(all_lows[i - window:i + window + 1]):
            support_levels.append(float(all_lows[i]))

    # Also add period high/low
    resistance_levels.append(float(all_highs.max()))
    support_levels.append(float(all_lows.min()))

    # Cluster nearby levels (within 1% of each other)
    def cluster_levels(levels, threshold_pct=1.0):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < threshold_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # Return mean of each cluster, sorted by how many times it was touched
        return sorted(
            [(round(np.mean(c), 2), len(c)) for c in clusters],
            key=lambda x: -x[1],
        )

    res_clustered = cluster_levels(resistance_levels)
    sup_clustered = cluster_levels(support_levels)

    # Filter: resistance above current price, support below
    resistances = [{"level": lvl, "strength": count}
                   for lvl, count in res_clustered if lvl > current_price][:n_levels]
    supports = [{"level": lvl, "strength": count}
                for lvl, count in sup_clustered if lvl < current_price][:n_levels]

    # Sort resistance ascending, support descending
    resistances.sort(key=lambda x: x["level"])
    supports.sort(key=lambda x: -x["level"])

    return {
        "current_price": round(current_price, 2),
        "resistances": resistances,
        "supports": supports,
    }


@router.get("/support-resistance/{ticker}")
def get_support_resistance(
    ticker: str,
    period: str = Query("3mo", description="1mo, 3mo, 6mo, 1y"),
    n_levels: int = Query(3, description="Number of S/R levels to return"),
):
    """Calculate support and resistance levels for a ticker."""
    symbol = normalize_ticker(ticker)
    t = yf.Ticker(symbol)
    hist = t.history(period=period)
    if hist.empty:
        return {"error": f"Insufficient data for {symbol}"}
    hist = hist.dropna(subset=["Close"])

    if len(hist) < 10:
        return {"error": f"Insufficient data for {symbol}"}

    highs = hist["High"].tolist()
    lows = hist["Low"].tolist()
    closes = hist["Close"].tolist()

    sr = _find_support_resistance(highs, lows, closes, n_levels)
    pivots = _calculate_pivot_points(highs[-1], lows[-1], closes[-1])

    # Also calculate from last N-day high/low
    period_high = round(max(highs), 2)
    period_low = round(min(lows), 2)

    return {
        "ticker": symbol,
        "period": period,
        "data_points": len(hist),
        "current_price": sr["current_price"],
        "period_high": period_high,
        "period_low": period_low,
        "support_resistance": sr,
        "pivot_points": pivots,
        "analysis": {
            "nearest_support": sr["supports"][0]["level"] if sr["supports"] else None,
            "nearest_resistance": sr["resistances"][0]["level"] if sr["resistances"] else None,
            "distance_to_support_pct": round(
                (sr["current_price"] - sr["supports"][0]["level"]) / sr["current_price"] * 100, 2
            ) if sr["supports"] else None,
            "distance_to_resistance_pct": round(
                (sr["resistances"][0]["level"] - sr["current_price"]) / sr["current_price"] * 100, 2
            ) if sr["resistances"] else None,
        },
    }


@router.get("/pivot-points/{ticker}")
def get_pivot_points(ticker: str):
    """Calculate daily pivot points (based on previous session)."""
    symbol = normalize_ticker(ticker)
    t = yf.Ticker(symbol)
    hist = t.history(period="5d")
    if hist.empty:
        return {"error": f"Insufficient data for {symbol}"}
    hist = hist.dropna(subset=["Close"])

    if len(hist) < 2:
        return {"error": f"Insufficient data for {symbol}"}

    # Use previous session for pivot calculation
    prev = hist.iloc[-2]
    current_close = hist.iloc[-1]["Close"]

    pivots = _calculate_pivot_points(prev["High"], prev["Low"], prev["Close"])

    return {
        "ticker": symbol,
        "current_price": round(current_close, 2),
        "based_on": {
            "date": hist.index[-2].strftime("%Y-%m-%d"),
            "high": round(prev["High"], 2),
            "low": round(prev["Low"], 2),
            "close": round(prev["Close"], 2),
        },
        "pivot_points": pivots,
    }


# --- Cyclical Pattern Endpoints ---

@router.get("/cyclical/monthly/{ticker}")
def get_monthly_seasonality(
    ticker: str,
    years: int = Query(5, description="Years of history to analyze"),
):
    """Analyze monthly return patterns (seasonality). FREE — no AI cost."""
    return analyze_monthly_seasonality(ticker, years)


@router.get("/cyclical/day-of-week/{ticker}")
def get_day_of_week(
    ticker: str,
    months: int = Query(6, description="Months of data to analyze"),
):
    """Analyze day-of-week return patterns. FREE — no AI cost."""
    return analyze_day_of_week(ticker, months)


@router.get("/cyclical/sector-rotation")
def get_sector_rotation(
    months: int = Query(3, description="Lookback period in months"),
):
    """Analyze sector rotation — which sectors are in favor. FREE — no AI cost."""
    return analyze_sector_rotation(months)


@router.post("/cyclical/backtest-seasonal")
def run_seasonal_backtest(
    ticker: str = Query(...),
    buy_months: str = Query(..., description="Comma-separated buy months (1-12)"),
    sell_months: str = Query(..., description="Comma-separated sell months (1-12)"),
    years: int = Query(5),
):
    """Backtest a seasonal buy/sell strategy. FREE — no AI cost, pure price math."""
    buy = [int(m.strip()) for m in buy_months.split(",")]
    sell = [int(m.strip()) for m in sell_months.split(",")]
    return backtest_seasonal_strategy(ticker, buy, sell, years)
