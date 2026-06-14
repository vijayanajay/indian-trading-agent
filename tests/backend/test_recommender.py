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
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
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
    assert gap_signals[0]["direction"] == "BEARISH"

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
    assert gap_signals[0]["direction"] == "BEARISH"

    # 4. Test Gap Down (Filled): gap_pct <= -2.0, high >= prev_close
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
    gap_signals = [s for s in result["signals"] if "Gap Down (Filled)" in s["type"]]
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
    assert gap_signals[0]["direction"] == "BEARISH"


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
        mock_executor_instance.submit.return_value = mock_future
        with patch("backend.recommender.as_completed", return_value=[mock_future]):
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            
            res = recommend(universe="test", apply_market_bias=True, apply_event_filter=False, apply_concentration_check=False)
            
            assert len(res["strong_buys"]) == 1 or len(res["buys"]) == 1
            pick = res["strong_buys"][0] if res["strong_buys"] else res["buys"][0]
            
            # The filter adjustment should NOT be merged into signals, but stay in filter_adjustments
            signal_types = [s["type"] for s in pick["signals"]]
            assert "FII/DII Flow (BULLISH)" not in signal_types
            assert "Strong Uptrend" in signal_types
            
            # The filter_adjustments array should NOT be cleared (contain FII flow)
            assert len(pick["filter_adjustments"]) == 1
            assert "FII/DII Flow" in pick["filter_adjustments"][0]["type"]
            
            # The honest assessment fingerprint should contain both the base signals and the filter adjustments
            fp = pick["honest_assessment"]["fingerprint"]
            from backend.honest_assessment import compute_fingerprint
            expected_fp = compute_fingerprint(["Strong Uptrend", "FII/DII Flow (BULLISH)"], "UNKNOWN")
            assert fp == expected_fp


def test_get_event_filter_for_ticker():
    from backend.calendar_data import get_event_filter_for_ticker
    from datetime import date, timedelta

    with patch("backend.calendar_data.get_market_events_in_range") as mock_events, \
         patch("backend.calendar_data.get_earnings_in_range") as mock_earnings:

        mock_earnings.return_value = []

        # Test case 1: RBI policy tomorrow (1 day ahead)
        # Sensitive sectors: Banks, Finance, Financial Services, NBFCs
        today = date.today()
        tomorrow = today + timedelta(days=1)
        mock_events.return_value = [
            {"type": "RBI_POLICY", "date": tomorrow.strftime("%Y-%m-%d"), "name": "RBI Monetary Policy Decision"}
        ]

        # RELIANCE (Energy) -> non-sensitive -> reduced penalty (0.3 * -1.5 = -0.45)
        res_rel = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res_rel["has_event"] is True
        assert res_rel["score_adjustment"] == -0.45
        assert "RBI policy tomorrow" in res_rel["warning"]

        # SBIN (Banks) -> sensitive -> full penalty (-1.5)
        res_sbin = get_event_filter_for_ticker("SBIN", days_ahead=2)
        assert res_sbin["has_event"] is True
        assert res_sbin["score_adjustment"] == -1.5

        # Test case 2: FOMC in 2 days
        # Sensitive sectors: IT, Information Technology, Pharma, Chemicals
        two_days = today + timedelta(days=2)
        mock_events.return_value = [
            {"type": "FOMC", "date": two_days.strftime("%Y-%m-%d"), "name": "US Fed FOMC Meeting"}
        ]

        # RELIANCE (Energy) -> non-sensitive -> reduced penalty (0.3 * -1.0 = -0.3)
        res_rel = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res_rel["has_event"] is True
        assert res_rel["score_adjustment"] == -0.3
        assert "Fed FOMC in 2 days" in res_rel["warning"]

        # INFY (IT) -> sensitive -> full penalty (-1.0)
        res_infy = get_event_filter_for_ticker("INFY", days_ahead=2)
        assert res_infy["has_event"] is True
        assert res_infy["score_adjustment"] == -1.0

        # Test case 3: FOMC in 3 days (ignored because days_ahead defaults to 2)
        three_days = today + timedelta(days=3)
        mock_events.return_value = [
            {"type": "FOMC", "date": three_days.strftime("%Y-%m-%d"), "name": "US Fed FOMC Meeting"}
        ]
        res = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res["has_event"] is False
        assert res["score_adjustment"] == 0.0

        # Test case 4: BUDGET tomorrow (1 day ahead)
        # All sectors get full penalty (-2.0)
        mock_events.return_value = [
            {"type": "BUDGET", "date": tomorrow.strftime("%Y-%m-%d"), "name": "Union Budget"}
        ]

        # RELIANCE (Energy) -> full penalty (-2.0)
        res_rel = get_event_filter_for_ticker("RELIANCE", days_ahead=2)
        assert res_rel["has_event"] is True
        assert res_rel["score_adjustment"] == -2.0
        assert "Budget in 1 day(s)" in res_rel["warning"]

        # SUNPHARMA (Pharma) -> also full penalty (-2.0)
        res_sun = get_event_filter_for_ticker("SUNPHARMA", days_ahead=2)
        assert res_sun["has_event"] is True
        assert res_sun["score_adjustment"] == -2.0


def test_recompute_confidence_and_counts_in_filters():
    def get_base_res():
        return {
            "score": 3.0,
            "bullish_signal_count": 3,
            "bearish_signal_count": 0,
            "confidence": "MEDIUM",
            "signals": [
                {"type": "Volume Spike (Bullish)", "direction": "BULLISH", "value": "2.5x avg", "weight": 2.0},
                {"type": "Near Major Support", "direction": "BULLISH", "value": "1.0% above low", "weight": 2.0},
                {"type": "Strong Uptrend", "direction": "BULLISH", "value": "Price > 50 SMA > 200 SMA", "weight": 1.0},
            ],
            "filter_adjustments": []
        }

    # 1. Apply a bullish market bias flow (should inflate aligned count to 4 -> HIGH confidence)
    bias = {"bias": "BULLISH", "score_adjustment": 1.0, "reasoning": "FII buying"}
    out = _apply_market_bias(get_base_res(), bias)
    assert out["bullish_signal_count"] == 4
    assert out["bearish_signal_count"] == 0
    assert out["confidence"] == "HIGH"

    # 2. Apply a bearish market bias flow instead (aligned count remains 3 -> MEDIUM confidence, bearish count becomes 1)
    bias_bear = {"bias": "BEARISH", "score_adjustment": -1.0, "reasoning": "FII selling"}
    out_bear = _apply_market_bias(get_base_res(), bias_bear)
    assert out_bear["bullish_signal_count"] == 3
    assert out_bear["bearish_signal_count"] == 1
    assert out_bear["confidence"] == "MEDIUM"

    # 3. Apply a concentration filter (should add BEARISH count)
    conc = {"sector": "IT", "score_adjustment": -2.0, "warnings": ["High risk"]}
    out_conc = _apply_concentration_filter(get_base_res(), conc)
    assert out_conc["bullish_signal_count"] == 3
    assert out_conc["bearish_signal_count"] == 1
    assert out_conc["confidence"] == "MEDIUM"

    # 4. Apply an event filter (should add BEARISH count)
    event = {"has_event": True, "score_adjustment": -2.0, "warning": "RBI Policy"}
    out_event = _apply_event_filter(get_base_res(), event)
    assert out_event["bullish_signal_count"] == 3
    assert out_event["bearish_signal_count"] == 1
    assert out_event["confidence"] == "MEDIUM"

    # 5. Verify direct calling of _recompute_confidence_and_counts with include_filters=True
    from backend.recommender import _recompute_confidence_and_counts
    res_with_filter = get_base_res()
    res_with_filter["filter_adjustments"].append({"type": "FII/DII Flow (BULLISH)", "direction": "BULLISH", "value": "buying", "weight": 1.0})
    out_direct = _recompute_confidence_and_counts(res_with_filter, include_filters=True)
    assert out_direct["bullish_signal_count"] == 4
    assert out_direct["confidence"] == "HIGH"


def test_filter_updates_trade_plan():
    import numpy as np
    # Test that filter adjustments update stop loss, target price, and R:R ratio
    res = {
        "score": 3.0,
        "price": 100.0,
        "direction": "BUY",
        "signals": [],
        "filter_adjustments": [],
        "_highs": np.array([105.0] * 60),
        "_lows": np.array([95.0] * 60),
        "_closes": np.array([100.0] * 60),
        "suggested_stop_loss": 98.0,
        "target_price": 104.0,
        "risk_reward_ratio": 2.0,
    }

    # Case 1: Apply a large bearish bias that downgrades BUY -> NEUTRAL.
    # The trade plan levels should be cleared (None).
    bias = {"bias": "BEARISH", "score_adjustment": -4.0, "reasoning": "FII selling"}
    with patch("backend.honest_assessment.get_honest_assessment") as mock_honest:
        # Return probability resulting in NEUTRAL direction (e.g. 50%)
        mock_honest.return_value = {"probability": 50.0}
        out = _apply_market_bias(res.copy(), bias)
        assert out["direction"] == "NEUTRAL"
        assert out["suggested_stop_loss"] is None
        assert out["target_price"] is None
        assert out["risk_reward_ratio"] is None

    # Case 2: Apply a bearish bias that flips BUY -> SELL (SHORT trade).
    # Since we don't have support/resistance mocks, it should run the short levels calculations.
    # We patch _find_support_resistance to return mock support/resistance
    with patch("backend.honest_assessment.get_honest_assessment") as mock_honest, \
         patch("backend.routers.strategies._find_support_resistance") as mock_sr:
        
        mock_honest.return_value = {"probability": 30.0}  # Will map to SELL (SHORT)
        mock_sr.return_value = {"supports": [{"level": 90.0}], "resistances": [{"level": 102.0}]}
        
        out = _apply_market_bias(res.copy(), bias)
        assert out["direction"] == "STRONG SELL"
        # For a short trade:
        # resistance is 102.0, fallback_dist is min(0.02 * 100, 2 * atr)
        # atr for all flat elements is 0. fallback_dist is 0.02 * 100 = 2.0
        # Since 102.0 - 100.0 = 2.0 <= 2.0, suggested_stop_loss is nearest resistance = 102.0.
        # target_price is nearest support = 90.0.
        # risk = 102 - 100 = 2.0. reward = 100 - 90 = 10.0. R:R = 10.0 / 2.0 = 5.0.
        assert out["suggested_stop_loss"] == 102.0
        assert out["target_price"] == 90.0
        assert out["risk_reward_ratio"] == 5.0


@patch("backend.recommender.UNIVERSES")
@patch("backend.recommender._refresh_active_weights")
@patch("backend.recommender.ThreadPoolExecutor")
@patch("backend.shadow_trades.record_shadow_trades_from_recommendations")
def test_recommend_failed_tickers(mock_shadow, mock_executor, mock_refresh, mock_universes):
    mock_universes.get.return_value = ["AAPL", "MSFT", "GOOGL"]

    # AAPL succeeds
    mock_future1 = MagicMock()
    mock_future1.result.return_value = {"ticker": "AAPL", "score": 5.0, "direction": "STRONG BUY", "bullish_signal_count": 3, "bearish_signal_count": 0}

    # MSFT fails (returns None)
    mock_future2 = MagicMock()
    mock_future2.result.return_value = None

    # GOOGL raises exception
    mock_future3 = MagicMock()
    mock_future3.result.side_effect = Exception("yfinance error")

    mock_executor_instance = MagicMock()
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, mock_future3]
    with patch("backend.recommender.as_completed", return_value=[mock_future1, mock_future2, mock_future3]):
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        res = recommend(universe="test", apply_market_bias=False, apply_event_filter=False, apply_concentration_check=False)

        assert len(res["strong_buys"]) == 1
        assert res["strong_buys"][0]["ticker"] == "AAPL"
        # Both MSFT and GOOGL should be in failed_tickers
        assert "MSFT" in res["failed_tickers"]
        assert "GOOGL" in res["failed_tickers"]
        assert len(res["failed_tickers"]) == 2



