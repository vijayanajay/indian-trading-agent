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
    res = {"score": 3.0, "bullish_signal_count": 2, "bearish_signal_count": 0, "signals": []}
    bias = {"bias": "BULLISH", "score_adjustment": 1.0, "reasoning": "buying"}

    out = _apply_market_bias(res, bias)
    assert out["score"] == 4.0
    assert out["direction"] == "STRONG BUY"
    assert out["market_bias_applied"] == "BULLISH"
    assert out["market_bias_score_adj"] == 1.0
    assert "FII/DII Flow" in out["filter_adjustments"][0]["type"]
    assert len(out["signals"]) == 0

def test_apply_concentration_filter():
    res = {"score": 3.0, "bullish_signal_count": 2, "signals": []}
    conc = {"sector": "IT", "score_adjustment": -2.0, "warnings": ["High risk"]}

    out = _apply_concentration_filter(res, conc)
    assert out["score"] == 1.0
    assert out["direction"] == "NEUTRAL"
    assert "Concentration" in out["filter_adjustments"][0]["type"]
    assert len(out["signals"]) == 0

def test_apply_event_filter():
    res = {"score": 3.0, "bullish_signal_count": 2, "signals": []}
    event = {"has_event": True, "score_adjustment": -2.0, "warning": "RBI Policy"}

    out = _apply_event_filter(res, event)
    assert out["score"] == 1.0
    assert out["direction"] == "NEUTRAL"
    assert "Event Risk" in out["filter_adjustments"][0]["type"]
    assert len(out["signals"]) == 0

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

    # 3. Test Gap Up (Unfilled - Green Candle - Fade): gap_pct >= 2.0, low > prev_close, close >= open
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
    gap_signals = [s for s in result["signals"] if "Gap Up (Unfilled)" in s["type"]]
    assert len(gap_signals) == 1
    assert gap_signals[0]["direction"] == "FADE"

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


def test_filter_adjustments_merged_and_hashed():
    # Test that recommend() correctly applies filters, merges adjustments into signals,
    # and recomputes get_honest_assessment with the merged signals.
    from backend.recommender import recommend
    
    with patch("backend.recommender.UNIVERSES") as mock_universes, \
         patch("backend.recommender._refresh_active_weights") as mock_refresh, \
         patch("backend.recommender.ThreadPoolExecutor") as mock_executor, \
         patch("backend.fii_dii.get_market_bias") as mock_bias, \
         patch("backend.shadow_trades.record_shadow_trades_from_recommendations") as mock_shadow:
         
        mock_universes.get.return_value = ["AAPL"]
        mock_bias.return_value = {
            "bias": "BULLISH",
            "score_adjustment": 1.5,
            "reasoning": "FII buying",
            "today_fii_net": 2500,
            "today_dii_net": 500,
        }
        
        # Mock analyze_stock to return a mock pick
        mock_future = MagicMock()
        mock_future.result.return_value = {
            "ticker": "AAPL",
            "symbol": "AAPL.NS",
            "price": 150.0,
            "change_pct": 1.2,
            "rsi": 45.0,
            "score": 3.0,
            "direction": "BUY",
            "confidence": "HIGH",
            "signals": [{"type": "Strong Uptrend", "direction": "BULLISH", "value": "Price > 50 SMA > 200 SMA", "weight": 1.0}],
            "filter_adjustments": [],
            "bullish_signal_count": 2,
            "bearish_signal_count": 0,
            "near_support": 145.0,
            "near_resistance": 155.0,
        }
        
        mock_executor_instance = MagicMock()
        with patch("backend.recommender.as_completed", return_value=[mock_future]):
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            
            res = recommend(universe="test", apply_market_bias=True, apply_event_filter=False, apply_concentration_check=False)
            
            assert len(res["strong_buys"]) == 1 or len(res["buys"]) == 1
            pick = res["strong_buys"][0] if res["strong_buys"] else res["buys"][0]
            
            # The filter adjustment should be merged into signals
            signal_types = [s["type"] for s in pick["signals"]]
            assert "FII/DII Flow (BULLISH)" in signal_types
            assert "Strong Uptrend" in signal_types
            
            # The filter_adjustments array should be cleared to prevent UI duplication
            assert len(pick["filter_adjustments"]) == 0
            
            # The honest assessment fingerprint should contain the FII flow signal
            fp = pick["honest_assessment"]["fingerprint"]
            from backend.honest_assessment import compute_fingerprint
            expected_fp = compute_fingerprint(["FII/DII Flow (BULLISH)", "Strong Uptrend"], "UNKNOWN")
            assert fp == expected_fp


def test_get_event_filter_for_ticker():
    from backend.calendar_data import get_event_filter_for_ticker
    from datetime import date, timedelta

    with patch("backend.calendar_data.get_market_events_in_range") as mock_events, \
         patch("backend.calendar_data.get_earnings_in_range") as mock_earnings:

        mock_earnings.return_value = []

        # Test case 1: RBI policy tomorrow (1 day ahead)
        today = date.today()
        tomorrow = today + timedelta(days=1)
        mock_events.return_value = [
            {"type": "RBI_POLICY", "date": tomorrow.strftime("%Y-%m-%d"), "name": "RBI Monetary Policy Decision"}
        ]

        res = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res["has_event"] is True
        assert res["score_adjustment"] == -1.5
        assert "RBI policy tomorrow" in res["warning"]

        # Test case 2: FOMC in 2 days
        two_days = today + timedelta(days=2)
        mock_events.return_value = [
            {"type": "FOMC", "date": two_days.strftime("%Y-%m-%d"), "name": "US Fed FOMC Meeting"}
        ]

        res = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res["has_event"] is True
        assert res["score_adjustment"] == -1.0
        assert "Fed FOMC in 2 days" in res["warning"]

        # Test case 3: FOMC in 3 days (ignored because days_ahead defaults to 2)
        three_days = today + timedelta(days=3)
        mock_events.return_value = [
            {"type": "FOMC", "date": three_days.strftime("%Y-%m-%d"), "name": "US Fed FOMC Meeting"}
        ]
        res = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res["has_event"] is False
        assert res["score_adjustment"] == 0.0
