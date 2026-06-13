"""Sector Concentration API."""

from fastapi import APIRouter, Query
from backend.concentration import (
    get_sector_allocation,
    get_concentration_summary,
    check_new_trade_concentration,
    get_open_positions,
    get_sector_for_ticker,
    check_correlation_clustering,
)

router = APIRouter(prefix="/api/concentration", tags=["concentration"])


@router.get("/summary")
def summary(total_capital: float = Query(500000, gt=0), fetch_if_missing: bool = Query(True)):
    """High-level concentration summary for dashboard widget."""
    return get_concentration_summary(total_capital=total_capital, fetch_if_missing=fetch_if_missing)


@router.get("/correlation/{ticker}")
def get_correlation_check(
    ticker: str,
    total_capital: float = Query(500000, gt=0),
    fetch_if_missing: bool = Query(True),
):
    """Check correlation clustering risk for a ticker against open positions."""
    return check_correlation_clustering(ticker, total_capital=total_capital, fetch_if_missing=fetch_if_missing)



@router.get("/allocation")
def allocation(total_capital: float = Query(500000, gt=0)):
    """Detailed sector allocation across all open positions."""
    return get_sector_allocation(total_capital)


@router.get("/check/{ticker}")
def check(ticker: str, position_value: float = Query(50000, gt=0), total_capital: float = Query(500000, gt=0)):
    """Check if adding a new trade in this ticker would breach concentration limits."""
    return check_new_trade_concentration(ticker, position_value, total_capital)


@router.get("/positions")
def positions():
    """Get all current open positions across paper + real trades."""
    pos = get_open_positions()
    return {"positions": pos, "count": len(pos)}


@router.get("/sector/{ticker}")
def sector_for(ticker: str):
    """Get the sector for a ticker."""
    return {"ticker": ticker.upper(), "sector": get_sector_for_ticker(ticker)}
