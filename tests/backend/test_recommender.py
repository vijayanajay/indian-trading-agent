import pytest
from unittest.mock import patch, MagicMock
from backend.recommender import recommend, _apply_market_bias, _apply_concentration_filter, _apply_event_filter

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
