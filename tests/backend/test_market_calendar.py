import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from tradingagents.utils.market_calendar import count_trading_days, NSE_HOLIDAYS, is_trading_day
from backend.simulation import refresh_paper_trade_prices
from backend.shadow_trades import refresh_shadow_prices

def test_count_trading_days_basic():
    # 2026-06-08 (Monday) to 2026-06-09 (Tuesday) -> 1 trading day
    start = date(2026, 6, 8)
    end = date(2026, 6, 9)
    assert count_trading_days(start, end) == 1

def test_count_trading_days_weekend():
    # Friday 2026-06-05 to Monday 2026-06-08 -> 1 trading day (Sat/Sun excluded)
    start = date(2026, 6, 5)
    end = date(2026, 6, 8)
    assert count_trading_days(start, end) == 1

    # Friday 2026-06-05 to Saturday 2026-06-06 -> 0 trading days
    end_sat = date(2026, 6, 6)
    assert count_trading_days(start, end_sat) == 0

def test_count_trading_days_holiday():
    # Republic day 2026-01-26 (Monday) is in NSE_HOLIDAYS.
    # Friday 2026-01-23 to Tuesday 2026-01-27:
    # 23 (Fri) -> 24 (Sat, no) -> 25 (Sun, no) -> 26 (Mon, holiday) -> 27 (Tue, yes)
    # Total trading days elapsed by Tuesday is 1.
    start = date(2026, 1, 23)
    end = date(2026, 1, 27)
    assert count_trading_days(start, end) == 1

def test_count_trading_days_same_day():
    start = date(2026, 6, 8)
    assert count_trading_days(start, start) == 0

def test_count_trading_days_invalid_range():
    # start after end
    start = date(2026, 6, 9)
    end = date(2026, 6, 8)
    assert count_trading_days(start, end) == 0

def test_count_trading_days_datetime_inputs():
    start = datetime(2026, 6, 8, 10, 0)
    end = datetime(2026, 6, 9, 15, 0)
    assert count_trading_days(start, end) == 1


@patch("backend.simulation.list_paper_trades")
@patch("backend.simulation.update_paper_trade_prices")
@patch("backend.simulation.update_paper_trade_status")
@patch("backend.simulation._price_n_days_later")
def test_refresh_paper_trade_prices_with_trading_days(
    mock_price_later, mock_update_status, mock_update_prices, mock_list_trades
):
    # Mocking date.today() to be Tuesday 2026-06-09
    # Friday 2026-06-05 is entry date (1 trading day elapsed)
    mock_list_trades.return_value = [
        {
            "id": 1,
            "ticker": "RELIANCE",
            "entry_date": "2026-06-05",
            "status": "active",
            "price_1d": None,
            "price_3d": None,
        }
    ]
    mock_price_later.return_value = 2500.0

    # With 1 trading day elapsed, it should fetch price_1d, but not price_3d
    with patch("backend.simulation.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 9) # Tuesday
        # We also need mock_date to inherit date methods for datetime parsing
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
        
        refresh_paper_trade_prices(trade_id=1)

    mock_update_prices.assert_called_once_with(1, {"price_1d": 2500.0})
    # Since only 1 trading day elapsed, it shouldn't auto-expire (requires > 10 trading days)
    mock_update_status.assert_not_called()


@patch("backend.shadow_trades.get_db")
@patch("backend.simulation._price_n_days_later")
def test_refresh_shadow_prices_with_trading_days(mock_price_later, mock_get_db):
    # Setup mock cursor and DB query
    mock_conn = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn

    # Friday 2026-06-05 entry date
    # Tuesday 2026-06-09 is today (1 trading day elapsed)
    mock_conn.execute.return_value.fetchall.return_value = [
        {
            "ticker": "TCS",
            "signal_date": "2026-06-05",
            "entry_price": 3000.0,
            "price_1d": None,
            "price_3d": None,
            "price_5d": None,
            "price_10d": None,
        }
    ]
    mock_price_later.return_value = 3100.0

    with patch("backend.shadow_trades.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 9) # Tuesday
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
        
        refresh_shadow_prices()

    # Verify that only price_1d update was prepared since trading_days_elapsed is 1
    # Check if execute was called to update the shadow trade
    update_calls = [
        call for call in mock_conn.execute.call_args_list 
        if "UPDATE shadow_trades" in call[0][0]
    ]
    assert len(update_calls) == 1
    query, params = update_calls[0][0]
    assert "price_1d = ?" in query
    assert "price_3d" not in query
