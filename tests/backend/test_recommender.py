import pytest
from unittest.mock import patch, MagicMock
from backend.recommender import recommend, _apply_market_bias, _apply_concentration_filter, _apply_event_filter

@pytest.fixture(autouse=True)
def clean_db_and_cache():
    from backend.db import get_db
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM model_coefficients")
    except Exception:
        pass
    import backend.signal_model
    backend.signal_model._MODEL_CACHE = None
    yield

def test_apply_market_bias():
    res = {"score": 3.0, "bullish_signal_count": 2, "bearish_signal_count": 0}
    bias = {"bias": "BULLISH", "score_adjustment": 1.0, "reasoning": "buying"}

    out = _apply_market_bias(res, bias)
    assert out["score"] == 4.0
    assert out["direction"] == "STRONG BUY"
    assert out["market_bias_applied"] == "BULLISH"

def test_apply_concentration_filter():
    res = {"score": 3.0, "bullish_signal_count": 2}
    conc = {"sector": "IT", "score_adjustment": -2.0, "warnings": ["High risk"]}

    out = _apply_concentration_filter(res, conc)
    assert out["score"] == 1.0
    assert out["direction"] == "NEUTRAL"
    assert "Concentration" in out["signals"][0]["type"]

def test_apply_event_filter():
    res = {"score": 3.0, "bullish_signal_count": 2}
    event = {"has_event": True, "score_adjustment": -2.0, "warning": "RBI Policy"}

    out = _apply_event_filter(res, event)
    assert out["score"] == 1.0
    assert out["direction"] == "NEUTRAL"
    assert "Event Risk" in out["signals"][0]["type"]

@patch("backend.recommender.UNIVERSES")
@patch("backend.recommender._refresh_active_weights")
@patch("backend.recommender.ThreadPoolExecutor")
@patch("backend.shadow_trades.record_shadow_trades_from_recommendations")
def test_recommend_base(mock_shadow, mock_executor, mock_refresh, mock_universes):
    mock_universes.get.return_value = ["AAPL", "MSFT"]

    # Mock futures
    mock_future1 = MagicMock()
    mock_future1.result.return_value = {"ticker": "AAPL", "score": 5.0, "direction": "STRONG BUY", "bullish_signal_count": 3, "bearish_signal_count": 0}

    mock_future2 = MagicMock()
    mock_future2.result.return_value = {"ticker": "MSFT", "score": -5.0, "direction": "STRONG SELL", "bullish_signal_count": 0, "bearish_signal_count": 3}

    mock_executor_instance = MagicMock()
    # patch as_completed to yield our futures
    with patch("backend.recommender.as_completed", return_value=[mock_future1, mock_future2]):
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        res = recommend(universe="test", apply_market_bias=False, apply_event_filter=False, apply_concentration_check=False)

        assert len(res["strong_buys"]) == 1
        assert res["strong_buys"][0]["ticker"] == "AAPL"
        assert len(res["strong_sells"]) == 1
        assert res["strong_sells"][0]["ticker"] == "MSFT"


@patch("backend.recommender.yf.Ticker")
def test_gap_signals(mock_ticker):
    import pandas as pd
    import numpy as np
    from backend.recommender import _analyze_stock

    dates = pd.date_range(end="2026-06-07", periods=60)
    volumes = [1000] * 60
    
    # 1. Test Gap Up (Filled): gap_pct >= 2.0, low <= prev_close
    # prev_close = 100, open = 103 (gap 3%), low = 99 (filled), close = 102
    opens = [100.0] * 59 + [103.0]
    highs = [101.0] * 59 + [104.0]
    lows = [99.0] * 59 + [99.0]
    closes = [100.0] * 59 + [102.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"gap": True})
    assert result is not None
    gap_signals = [s for s in result["signals"] if "Gap Up (Filled)" in s["type"]]
    assert len(gap_signals) == 1
    assert gap_signals[0]["direction"] == "BULLISH"

    # 2. Test Gap Up (Unfilled - Fade): gap_pct >= 2.0, low > prev_close, close < open
    # prev_close = 100, open = 103 (gap 3%), low = 101 (unfilled), close = 102 (close < open)
    opens = [100.0] * 59 + [103.0]
    highs = [104.0] * 59 + [104.0]
    lows = [99.0] * 59 + [101.0]
    closes = [100.0] * 59 + [102.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"gap": True})
    assert result is not None
    gap_signals = [s for s in result["signals"] if "Gap Up (Unfilled)" in s["type"]]
    assert len(gap_signals) == 1
    assert gap_signals[0]["direction"] == "FADE"

    # 3. Test Gap Up (Unfilled - No Signal): gap_pct >= 2.0, low > prev_close, close >= open
    # prev_close = 100, open = 103 (gap 3%), low = 101 (unfilled), close = 104 (close >= open)
    opens = [100.0] * 59 + [103.0]
    highs = [105.0] * 59 + [105.0]
    lows = [99.0] * 59 + [101.0]
    closes = [100.0] * 59 + [104.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"gap": True})
    assert result is not None
    gap_signals = [s for s in result["signals"] if "Gap" in s["type"]]
    assert len(gap_signals) == 0

    # 4. Test Gap Down (Filled - Reversal): gap_pct <= -2.0, high >= prev_close
    # prev_close = 100, open = 97 (gap -3%), high = 100.5 (filled), close = 99
    opens = [100.0] * 59 + [97.0]
    highs = [101.0] * 59 + [100.5]
    lows = [99.0] * 59 + [96.0]
    closes = [100.0] * 59 + [99.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"gap": True})
    assert result is not None
    gap_signals = [s for s in result["signals"] if "Gap Down (Filled - Reversal)" in s["type"]]
    assert len(gap_signals) == 1
    assert gap_signals[0]["direction"] == "BULLISH"

    # 5. Test Gap Down (Unfilled): gap_pct <= -2.0, high < prev_close
    # prev_close = 100, open = 97 (gap -3%), high = 99 (unfilled), close = 98
    opens = [100.0] * 59 + [97.0]
    highs = [101.0] * 59 + [99.0]
    lows = [99.0] * 59 + [96.0]
    closes = [100.0] * 59 + [98.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"gap": True})
    assert result is not None
    gap_signals = [s for s in result["signals"] if "Gap Down (Unfilled)" in s["type"]]
    assert len(gap_signals) == 1
    assert gap_signals[0]["direction"] == "FADE"
