import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timezone, timedelta
from dateutil.tz import tzutc

from tradingagents.dataflows.y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    _get_stock_stats_bulk,
    get_stockstats_indicator,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions
)

# Dummy test data generators
def create_ohlcv_data():
    dates = pd.date_range("2023-01-01", periods=5)
    return pd.DataFrame({
        "Date": dates,
        "Open": [10.123, 11.234, 12.345, 13.456, 14.567],
        "High": [11.123, 12.234, 13.345, 14.456, 15.567],
        "Low": [9.123, 10.234, 11.345, 12.456, 13.567],
        "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
        "Volume": [1000, 1100, 1200, 1300, 1400],
        "Adj Close": [10.5, 11.5, 12.5, 13.5, 14.5]
    }).set_index("Date")

def create_ohlcv_data_tz():
    dates = pd.date_range("2023-01-01", periods=2, tz=tzutc())
    return pd.DataFrame({
        "Date": dates,
        "Open": [10, 11],
        "High": [11, 12],
        "Low": [9, 10],
        "Close": [10.5, 11.5],
        "Volume": [1000, 1100],
        "Adj Close": [10.5, 11.5]
    }).set_index("Date")

# --- get_YFin_data_online tests ---

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_YFin_data_online_standard(mock_ticker):
    mock_history = MagicMock()
    mock_history.return_value = create_ohlcv_data()
    mock_ticker.return_value.history = mock_history

    res = get_YFin_data_online("AAPL", "2023-01-01", "2023-01-05")

    assert "# Stock data for AAPL from 2023-01-01 to 2023-01-05" in res
    assert "# Total records: 5" in res
    assert "Date,Open,High,Low,Close,Volume,Adj Close" in res
    assert "2023-01-01" in res
    assert "10.12" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_YFin_data_online_empty(mock_ticker):
    mock_history = MagicMock()
    mock_history.return_value = pd.DataFrame()
    mock_ticker.return_value.history = mock_history

    res = get_YFin_data_online("AAPL", "2023-01-01", "2023-01-05")
    assert "No data found for symbol 'AAPL' between 2023-01-01 and 2023-01-05" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_YFin_data_online_timezone(mock_ticker):
    mock_history = MagicMock()
    mock_history.return_value = create_ohlcv_data_tz()
    mock_ticker.return_value.history = mock_history

    res = get_YFin_data_online("AAPL", "2023-01-01", "2023-01-02")
    # TZ info should be removed, check format
    assert "2023-01-01" in res
    assert "10.5" in res

def test_get_YFin_data_online_invalid_date():
    with pytest.raises(ValueError):
        get_YFin_data_online("AAPL", "01-01-2023", "2023-01-05")

# --- _get_stock_stats_bulk tests ---

@patch("tradingagents.dataflows.y_finance.load_ohlcv")
def test__get_stock_stats_bulk_standard(mock_load_ohlcv):
    mock_load_ohlcv.return_value = create_ohlcv_data().reset_index()
    res = _get_stock_stats_bulk("AAPL", "close", "2023-01-05")

    assert res["2023-01-01"] == "10.5"
    assert res["2023-01-05"] == "14.5"

@patch("tradingagents.dataflows.y_finance.load_ohlcv")
def test__get_stock_stats_bulk_nan_values(mock_load_ohlcv):
    data = create_ohlcv_data().reset_index()
    data.loc[0, "Close"] = np.nan
    mock_load_ohlcv.return_value = data
    res = _get_stock_stats_bulk("AAPL", "close", "2023-01-05")

    assert res["2023-01-01"] == "N/A"

# --- get_stockstats_indicator tests ---

@patch("tradingagents.dataflows.y_finance.StockstatsUtils.get_stock_stats")
def test_get_stockstats_indicator_standard(mock_get_stats):
    mock_get_stats.return_value = 10.5
    res = get_stockstats_indicator("AAPL", "macd", "2023-01-05")
    assert res == "10.5"

@patch("tradingagents.dataflows.y_finance.StockstatsUtils.get_stock_stats")
def test_get_stockstats_indicator_exception(mock_get_stats):
    mock_get_stats.side_effect = Exception("Test Error")
    res = get_stockstats_indicator("AAPL", "macd", "2023-01-05")
    assert res == ""

# --- get_stock_stats_indicators_window tests ---

@patch("tradingagents.dataflows.y_finance._get_stock_stats_bulk")
def test_get_stock_stats_indicators_window_bulk_success(mock_bulk):
    mock_bulk.return_value = {
        "2023-01-05": "14.5",
        "2023-01-04": "13.5",
        "2023-01-03": "12.5",
    }
    res = get_stock_stats_indicators_window("AAPL", "macd", "2023-01-05", 3)

    assert "## macd values from 2023-01-02 to 2023-01-05:" in res
    assert "2023-01-05: 14.5" in res
    assert "2023-01-04: 13.5" in res
    assert "2023-01-03: 12.5" in res
    assert "2023-01-02: N/A" in res
    # assert MACD description

@patch("tradingagents.dataflows.y_finance._get_stock_stats_bulk")
@patch("tradingagents.dataflows.y_finance.get_stockstats_indicator")
def test_get_stock_stats_indicators_window_fallback(mock_get_indicator, mock_bulk):
    mock_bulk.side_effect = Exception("Bulk Error")
    mock_get_indicator.side_effect = lambda sym, ind, dt: "10.5" if dt == "2023-01-05" else "N/A"

    res = get_stock_stats_indicators_window("AAPL", "macd", "2023-01-05", 1)

    assert "2023-01-05: 10.5" in res

# --- get_fundamentals tests ---

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_fundamentals_standard(mock_ticker):
    mock_info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "trailingPE": 25.5,
        "missingField": "should not appear"
    }
    mock_ticker.return_value.info = mock_info

    res = get_fundamentals("AAPL")

    assert "# Company Fundamentals for AAPL" in res
    assert "Name: Apple Inc." in res
    assert "Sector: Technology" in res
    assert "PE Ratio (TTM): 25.5" in res
    assert "missingField" not in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_fundamentals_empty(mock_ticker):
    mock_ticker.return_value.info = {}
    res = get_fundamentals("AAPL")
    assert "No fundamentals data found for symbol 'AAPL'" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker.info", new_callable=PropertyMock)
def test_get_fundamentals_exception(mock_prop):
    mock_prop.side_effect = Exception("Test API Error")
    res = get_fundamentals("AAPL")
    assert "Error retrieving fundamentals for AAPL: Test API Error" in res


# --- Financial Statements tests (balance sheet, cashflow, income stmt) ---

def create_financials_data():
    dates = pd.date_range("2022-12-31", periods=2, freq="YE")
    return pd.DataFrame({
        "Item1": [100, 200],
        "Item2": [300, 400]
    }, index=dates).T

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_balance_sheet_standard(mock_ticker):
    mock_ticker.return_value.quarterly_balance_sheet = create_financials_data()
    mock_ticker.return_value.balance_sheet = create_financials_data() * 2

    res_q = get_balance_sheet("AAPL", "quarterly", "2023-05-01")
    assert "# Balance Sheet data for AAPL (quarterly)" in res_q
    assert "Item1" in res_q

    res_a = get_balance_sheet("AAPL", "annual", "2023-05-01")
    assert "# Balance Sheet data for AAPL (annual)" in res_a

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_balance_sheet_empty(mock_ticker):
    mock_ticker.return_value.quarterly_balance_sheet = pd.DataFrame()
    res = get_balance_sheet("AAPL")
    assert "No balance sheet data found for symbol 'AAPL'" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker.quarterly_balance_sheet", new_callable=PropertyMock)
def test_get_balance_sheet_exception(mock_prop):
    mock_prop.side_effect = Exception("API Error")
    res = get_balance_sheet("AAPL")
    assert "Error retrieving balance sheet for AAPL: API Error" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_cashflow_standard(mock_ticker):
    mock_ticker.return_value.quarterly_cashflow = create_financials_data()
    res = get_cashflow("AAPL", "quarterly", "2023-05-01")
    assert "# Cash Flow data for AAPL (quarterly)" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_cashflow_empty(mock_ticker):
    mock_ticker.return_value.quarterly_cashflow = pd.DataFrame()
    res = get_cashflow("AAPL")
    assert "No cash flow data found for symbol 'AAPL'" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_insider_transactions_standard(mock_ticker):
    data = pd.DataFrame({"Date": ["2023-01-01"], "Transaction": ["Buy"]})
    mock_ticker.return_value.insider_transactions = data
    res = get_insider_transactions("AAPL")
    assert "# Insider Transactions data for AAPL" in res
    assert "Buy" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_insider_transactions_empty(mock_ticker):
    mock_ticker.return_value.insider_transactions = pd.DataFrame()
    res = get_insider_transactions("AAPL")
    assert "No insider transactions data found for symbol 'AAPL'" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker")
def test_get_insider_transactions_none(mock_ticker):
    mock_ticker.return_value.insider_transactions = None
    res = get_insider_transactions("AAPL")
    assert "No insider transactions data found for symbol 'AAPL'" in res

@patch("tradingagents.dataflows.y_finance.yf.Ticker.insider_transactions", new_callable=PropertyMock)
def test_get_insider_transactions_exception(mock_prop):
    mock_prop.side_effect = Exception("API Error")
    res = get_insider_transactions("AAPL")
    assert "Error retrieving insider transactions for AAPL: API Error" in res
