import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from backend.cron import update_price_cache
from backend.db import get_db, ensure_db

@pytest.fixture(autouse=True)
def setup_db():
    ensure_db()
    # Clean stock_prices table before each test
    with get_db() as conn:
        conn.execute("DELETE FROM stock_prices")
        conn.commit()

@patch("yfinance.download")
def test_update_price_cache_multi_index(mock_download):
    # Simulate yfinance returning a MultiIndex DataFrame for a multi-ticker chunk
    dates = pd.date_range("2026-06-01", periods=5)
    cols = pd.MultiIndex.from_tuples([
        ("AAPL.NS", "Open"), ("AAPL.NS", "High"), ("AAPL.NS", "Low"), ("AAPL.NS", "Close"), ("AAPL.NS", "Volume"),
        ("MSFT.NS", "Open"), ("MSFT.NS", "High"), ("MSFT.NS", "Low"), ("MSFT.NS", "Close"), ("MSFT.NS", "Volume"),
    ])
    data = [
        [150.0, 155.0, 149.0, 154.0, 1000.0, 300.0, 305.0, 299.0, 304.0, 2000.0],
        [154.0, 156.0, 153.0, 155.0, 1100.0, 304.0, 308.0, 303.0, 307.0, 2100.0],
        [155.0, 158.0, 154.0, 157.0, 1200.0, 307.0, 310.0, 306.0, 309.0, 2200.0],
        [157.0, 159.0, 156.0, 158.0, 1300.0, 309.0, 312.0, 308.0, 311.0, 2300.0],
        [158.0, 160.0, 157.0, 159.0, 1400.0, 311.0, 315.0, 310.0, 314.0, 2400.0],
    ]
    df = pd.DataFrame(data, index=dates, columns=cols)
    mock_download.return_value = df

    res = update_price_cache(tickers_list=["AAPL", "MSFT"])
    
    assert res["success"] == 2
    assert res["failed"] == 0
    assert res["bars_saved"] == 10

    # Verify database contents
    with get_db() as conn:
        aapl_rows = conn.execute("SELECT * FROM stock_prices WHERE ticker = 'AAPL'").fetchall()
        msft_rows = conn.execute("SELECT * FROM stock_prices WHERE ticker = 'MSFT'").fetchall()
        assert len(aapl_rows) == 5
        assert len(msft_rows) == 5
        assert aapl_rows[0]["open"] == 150.0
        assert msft_rows[0]["open"] == 300.0

@patch("yfinance.download")
def test_update_price_cache_flat_index_single_ticker(mock_download):
    # Simulate yfinance returning a flat DataFrame for a single-ticker chunk
    dates = pd.date_range("2026-06-01", periods=5)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    data = [
        [150.0, 155.0, 149.0, 154.0, 1000.0],
        [154.0, 156.0, 153.0, 155.0, 1100.0],
        [155.0, 158.0, 154.0, 157.0, 1200.0],
        [157.0, 159.0, 156.0, 158.0, 1300.0],
        [158.0, 160.0, 157.0, 159.0, 1400.0],
    ]
    df = pd.DataFrame(data, index=dates, columns=cols)
    mock_download.return_value = df

    res = update_price_cache(tickers_list=["AAPL"])
    
    assert res["success"] == 1
    assert res["failed"] == 0
    assert res["bars_saved"] == 5

    # Verify database contents
    with get_db() as conn:
        aapl_rows = conn.execute("SELECT * FROM stock_prices WHERE ticker = 'AAPL'").fetchall()
        assert len(aapl_rows) == 5
        assert aapl_rows[0]["open"] == 150.0

@patch("yfinance.download")
def test_update_price_cache_flat_index_multi_ticker_warning(mock_download):
    # Simulate yfinance returning a flat DataFrame for a multi-ticker chunk
    dates = pd.date_range("2026-06-01", periods=5)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    data = [
        [150.0, 155.0, 149.0, 154.0, 1000.0],
        [154.0, 156.0, 153.0, 155.0, 1100.0],
        [155.0, 158.0, 154.0, 157.0, 1200.0],
        [157.0, 159.0, 156.0, 158.0, 1300.0],
        [158.0, 160.0, 157.0, 159.0, 1400.0],
    ]
    df = pd.DataFrame(data, index=dates, columns=cols)
    mock_download.return_value = df

    # We requested AAPL and MSFT but got a flat DataFrame.
    # This should be skipped to prevent writing AAPL prices under MSFT or vice-versa.
    res = update_price_cache(tickers_list=["AAPL", "MSFT"])
    
    assert res["success"] == 0
    assert res["failed"] == 2
    assert res["bars_saved"] == 0

    # Verify database is empty
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM stock_prices").fetchall()
        assert len(rows) == 0
