import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from backend.market_regime import classify_regime_for_date, get_current_regime, get_cached_regime, _annualized_vol

def create_mock_history(last_price, sma50_trend="flat", sma200_trend="flat", high_vol=False):
    # Need > 200 days. 250 is safe.
    dates = pd.date_range(end=date.today(), periods=250)

    # Base prices
    prices = np.ones(250) * 10000
    # Add tiny noise to avoid 0 vol
    prices += np.random.normal(0, 10, 250)

    if sma50_trend == "up":
        prices[-50:] = np.linspace(10000, 11000, 50)
    elif sma50_trend == "down":
        prices[-50:] = np.linspace(10000, 9000, 50)

    if sma200_trend == "up":
        prices = np.linspace(8000, prices[-1], 250)
    elif sma200_trend == "down":
        prices = np.linspace(12000, prices[-1], 250)

    if high_vol:
        # Inject massive volatility in the last 20 days
        noise = np.random.normal(0, 500, 20)
        prices[-20:] += noise



    df = pd.DataFrame({"Close": prices}, index=dates)
    return df

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_bull(mock_ticker):
    # Nifty > 50 SMA > 200 SMA
    # 200 SMA avg ~ 9000, 50 SMA avg ~ 10500, last price 11500
    df = create_mock_history(11500, "up", "up")
    mock_ticker.return_value.history.return_value = df

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "BULL"
    assert "Nifty 11000 > 50 SMA" in res["reasoning"]

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_bear(mock_ticker):
    # Nifty < 50 SMA < 200 SMA
    df = create_mock_history(8500, "down", "down")
    mock_ticker.return_value.history.return_value = df

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "BEAR"
    assert "Nifty 9000 < 50 SMA" in res["reasoning"]

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_sideways(mock_ticker):
    # Mixed trend: e.g. Nifty = 10000, SMA50 = 10000, SMA200 = 10000
    df = create_mock_history(10000, "flat", "flat")
    df.iloc[-50:, 0] = 10000 # SMA50 = 10000
    df.iloc[-200:-50, 0] = 10000 # SMA200 = 10000
    df.iloc[-1, 0] = 10000 # Nifty = 10000
    mock_ticker.return_value.history.return_value = df

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "SIDEWAYS"

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_sideways_mixed(mock_ticker):
    # Mixed trend not hugging 50 SMA
    df = create_mock_history(10000, "flat", "flat")
    # Make Nifty = 11000, SMA50 = 10000, SMA200 = 10500
    # to guarantee it's not BULL and not BEAR, and > 2% away from SMA 50
    df.iloc[-50:, 0] = 10000 # SMA50 = 10000
    df.iloc[-200:-50, 0] = 10666 # SMA200 ~ 10500
    df.iloc[-1, 0] = 11000 # Nifty = 11000 (10% away from 50 SMA)
    # prevent HIGH_VOL
    df.iloc[-20:, 0] = np.linspace(10000, 11000, 20)
    mock_ticker.return_value.history.return_value = df

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "SIDEWAYS"
    assert "Mixed trend" in res["reasoning"]

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_high_vol(mock_ticker):
    # Even if trend is BULL, high vol overrides it
    df = create_mock_history(11500, "up", "up", high_vol=True)
    # Ensure massive variance
    df.iloc[-2, 0] = 13000
    df.iloc[-3, 0] = 9000
    df.iloc[-4, 0] = 14000
    df.iloc[-5, 0] = 8000
    df.iloc[-6, 0] = 15000
    mock_ticker.return_value.history.return_value = df

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "HIGH_VOL"

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_exception(mock_ticker):
    mock_ticker.return_value.history.side_effect = Exception("API Error")
    res = classify_regime_for_date(date.today())
    assert res["regime"] == "UNKNOWN"
    assert "fetch failed" in res["reasoning"]

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_empty_data(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()

    res = classify_regime_for_date(date.today())
    assert res["regime"] == "UNKNOWN"
    assert "insufficient history" in res["reasoning"]

@patch("backend.market_regime.yf.Ticker")
def test_classify_regime_insufficient_before_target(mock_ticker):
    df = create_mock_history(11500)
    # filter leaves less than 200
    target_date = df.index[50].date()
    mock_ticker.return_value.history.return_value = df
    res = classify_regime_for_date(target_date)
    assert res["regime"] == "UNKNOWN"
    assert "insufficient history before target_date" in res["reasoning"]

@patch("backend.market_regime.classify_regime_for_date")
def test_get_current_regime(mock_classify):
    mock_classify.return_value = {"regime": "BULL"}
    res = get_current_regime()
    assert res["regime"] == "BULL"

@patch("backend.market_regime.classify_regime_for_date")
def test_get_cached_regime(mock_classify):
    mock_classify.return_value = {"regime": "BEAR"}
    d = date(2023, 1, 1)

    # Need to clear cache to be safe
    from backend.market_regime import _REGIME_CACHE
    if d.isoformat() in _REGIME_CACHE:
        del _REGIME_CACHE[d.isoformat()]

    # First call
    res1 = get_cached_regime(d)
    assert res1["regime"] == "BEAR"
    mock_classify.assert_called_once_with(d)

    # Second call uses cache
    res2 = get_cached_regime(d)
    assert res2["regime"] == "BEAR"
    assert mock_classify.call_count == 1

    # Ensure cache not populated on UNKNOWN
    d2 = date(2023, 1, 2)
    mock_classify.return_value = {"regime": "UNKNOWN"}
    get_cached_regime(d2)
    assert d2.isoformat() not in _REGIME_CACHE

def test_annualized_vol_short_array():
    res = _annualized_vol(np.array([1, 2, 3]), window=20)
    assert res == 0.0
