import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch
from backend.simulation import _analyze_stock_at_date

@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
def test_simulation_gap_up_filled(mock_ticker, mock_rsi, mock_get_regime):
    # Target date
    target_date = date(2026, 6, 8)
    
    # 70 days total: 60 past days + 10 future days
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    
    # Gap Up (Filled): gap_pct = 3%, low = 99 <= prev_close (100) -> should add 1.5
    # Target date index is 59 (60th element, matching target_date)
    target_idx = 59
    opens = [100.0] * 70
    highs = [101.0] * 70
    lows = [99.0] * 70
    closes = [100.0] * 70
    
    # Set the target date values
    opens[target_idx] = 103.0
    highs[target_idx] = 101.0
    lows[target_idx] = 99.0
    closes[target_idx] = 101.0
    
    # Volume spike: vol_ratio >= 2.0 and price_change > 0.5 -> score += 2.0
    volumes[target_idx] = 2000
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date)
    assert result is not None
    # Expected score: 1.5 (gap filled) + 2.0 (volume spike) = 3.5 (BUY)
    assert result["score"] == 3.5
    assert result["signal"] == "BUY"


@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
def test_simulation_gap_up_unfilled(mock_ticker, mock_rsi, mock_get_regime):
    target_date = date(2026, 6, 8)
    
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    
    # Gap Up (Unfilled): gap_pct = 3%, low = 100.5 > prev_close (100) -> should add -0.5
    target_idx = 59
    opens = [100.0] * 70
    highs = [101.0] * 70
    lows = [99.0] * 70
    closes = [100.0] * 70
    
    opens[target_idx] = 103.0
    highs[target_idx] = 101.0
    lows[target_idx] = 100.5
    closes[target_idx] = 99.0
    
    # Volume spike: vol_ratio >= 2.0 and price_change < -0.5 -> score += -2.0
    volumes[target_idx] = 2000
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date)
    assert result is not None
    # Expected score: -0.5 (gap unfilled) + -2.0 (volume spike bearish) = -2.5 (SELL)
    assert result["score"] == -2.5
    assert result["signal"] == "SELL"


@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
def test_simulation_gap_down_filled(mock_ticker, mock_rsi, mock_get_regime):
    target_date = date(2026, 6, 8)
    
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    
    # Gap Down (Filled): gap_pct = -3%, high = 100.5 >= prev_close (100) -> should add -1.5
    target_idx = 59
    opens = [100.0] * 70
    highs = [101.0] * 70
    lows = [96.0] * 70
    closes = [100.0] * 70
    
    opens[target_idx] = 97.0
    highs[target_idx] = 100.5
    lows[target_idx] = 96.0
    closes[target_idx] = 99.0
    
    # Volume spike: vol_ratio >= 2.0 and price_change < -0.5 -> score += -2.0
    volumes[target_idx] = 2000
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date)
    assert result is not None
    # Expected score: -1.5 (gap down filled) + -2.0 (volume spike bearish) = -3.5 (SELL)
    assert result["score"] == -3.5
    assert result["signal"] == "SELL"


@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
def test_simulation_gap_down_unfilled(mock_ticker, mock_rsi, mock_get_regime):
    target_date = date(2026, 6, 8)
    
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    
    # Gap Down (Unfilled): gap_pct = -3%, high = 99.0 < prev_close (100) -> should add -0.5
    target_idx = 59
    opens = [100.0] * 70
    highs = [101.0] * 70
    lows = [96.0] * 70
    closes = [100.0] * 70
    
    opens[target_idx] = 97.0
    highs[target_idx] = 99.0
    lows[target_idx] = 96.0
    closes[target_idx] = 98.0
    
    # Volume spike: vol_ratio >= 2.0 and price_change < -0.5 -> score += -2.0
    volumes[target_idx] = 2000
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date)
    assert result is not None
    # Expected score: -0.5 (gap down unfilled) + -2.0 (volume spike bearish) = -2.5 (SELL)
    assert result["score"] == -2.5
    assert result["signal"] == "SELL"


@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
@patch("backend.signal_performance.get_active_weights_for_regime")
def test_simulation_regime_weights(mock_get_weights, mock_ticker, mock_rsi):
    # Setup custom weights where breakout is extremely valued
    custom_weights = {
        "gap_up_filled": 1.5,
        "gap_up_open": -0.5,
        "gap_down_filled": 1.5,
        "gap_down_open": -0.5,
        "volume_bullish": 2.0,
        "volume_bearish": -2.0,
        "breakout_vol_confirmed": 10.0,  # normally 3.0
        "breakout_weak": 1.0,
        "near_support": 2.0,
        "near_resistance": -1.5,
        "breakdown_support": -2.5,
        "rsi_oversold": 1.5,
        "rsi_overbought": -1.0,
    }
    mock_get_weights.return_value = custom_weights
    
    target_date = date(2026, 6, 8)
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    opens = [100.0] * 70
    highs = [100.0] * 70
    lows = [99.0] * 70
    closes = [100.0] * 70
    
    # Setup breakout on target date
    target_idx = 59
    highs[target_idx] = 105.0  # above 20-day high (100)
    closes[target_idx] = 104.0
    volumes[target_idx] = 2000  # vol_ratio >= 1.5 (2.0x avg)
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date, regime="HIGH_VOL")
    
    mock_get_weights.assert_called_once_with("HIGH_VOL")
    assert result is not None
    # Expected score: 10.0 (custom breakout_vol_confirmed) + 2.0 (volume_bullish) = 12.0
    assert result["score"] == 12.0
    assert result["signal"] == "STRONG BUY"


@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
@patch("backend.simulation._compute_rsi", return_value=50.0)
@patch("backend.simulation.yf.Ticker")
def test_simulation_neutral_hold_signal(mock_ticker, mock_rsi, mock_get_regime):
    target_date = date(2026, 6, 8)
    
    dates = pd.date_range(end="2026-06-18", periods=70)
    volumes = [1000] * 70
    opens = [100.0] * 70
    highs = [101.0] * 70
    lows = [99.0] * 70
    closes = [100.0] * 70
    
    # Target date index is 59
    target_idx = 59
    opens[target_idx] = 100.0
    highs[target_idx] = 101.0
    lows[target_idx] = 99.0
    closes[target_idx] = 100.0
    
    # Future dates for returns
    # close after 5 trading days should be 105.0 (+5%)
    for i in range(target_idx + 1, 70):
        closes[i] = 105.0
        
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock_at_date("TEST", target_date)
    assert result is not None
    assert result["score"] == 0.0
    assert result["signal"] == "HOLD"
    assert result["return_5d"] == 5.0
    assert result["outcome_5d"] == "win"


@patch("backend.simulation.get_db")
@patch("backend.simulation.save_recommender_backtest_row")
@patch("backend.simulation._analyze_stock_at_date")
@patch("backend.market_regime.get_cached_regime", return_value={"regime": None})
def test_run_recommender_backtest_dates(mock_regime, mock_analyze, mock_save, mock_get_db):
    mock_analyze.return_value = {
        "ticker": "TEST",
        "signal": "BUY",
        "score": 3.0,
        "entry_price": 100.0,
        "return_5d": 5.0,
        "outcome_5d": "win",
        "confidence": "MEDIUM",
        "success_probability": None,
    }
    
    from backend.simulation import run_recommender_backtest
    result = run_recommender_backtest(
        universe="nifty50",
        start_date="2026-05-01",
        end_date="2026-06-01",
        interval_days=5,
    )
    
    # 2026-05-01 (Friday) -> skip 5 days -> 2026-05-06 (Wednesday) -> skip 5 days -> 2026-05-11 (Monday)
    # -> skip 5 days -> 2026-05-16 (Saturday -> Sunday -> Monday 2026-05-18)
    # -> skip 5 days -> 2026-05-23 (Saturday -> Sunday -> Monday 2026-05-25)
    # -> skip 5 days -> 2026-05-30 (Saturday -> Sunday -> Monday 2026-06-01)
    # Total dates tested: 6
    assert result["dates_tested"] == 6
