import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.performance import _summarize_trades
from backend.recommender import _analyze_stock

def test_risk_metrics_calculation():
    # Construct dummy trades with known returns to check metrics
    trades = [
        {"returns": {"day_5": 2.0}, "date": "2026-06-01"},
        {"returns": {"day_5": 4.0}, "date": "2026-06-02"},
        {"returns": {"day_5": -1.0}, "date": "2026-06-03"},
        {"returns": {"day_5": -3.0}, "date": "2026-06-04"},
        {"returns": {"day_5": 3.0}, "date": "2026-06-05"},
    ]

    # Patch DB call in performance to avoid writing to actual settings table during testing
    with patch("backend.db.set_setting") as mock_set:
        result = _summarize_trades(trades, [5], "Breakout Strategy")
        
        assert result["total_signals"] == 5
        assert not result["untradeable"]
        
        hp = result["hold_periods"]["day_5"]
        assert hp["avg_return"] == 1.0
        assert hp["best_trade"] == 4.0
        assert hp["worst_trade"] == -3.0
        assert hp["sharpe"] > 0
        assert hp["sortino"] > 0
        assert hp["max_drawdown"] == round((106.0 - 102.0) / 106.0 * 100.0, 2)
        assert hp["gain_to_pain"] == 2.25


def test_untradeable_flagging():
    # Sharpe/Sortino < 1.0 or Max DD > 15% makes it untradeable
    # returns: [-5.0, -10.0, -8.0, 2.0]
    # equity starting at 100: 100 -> 95 -> 85 -> 77 -> 79
    # peak = 100, min = 77, dd = 23% (which is > 15%)
    trades = [
        {"returns": {"day_5": -5.0}, "date": "2026-06-01"},
        {"returns": {"day_5": -10.0}, "date": "2026-06-02"},
        {"returns": {"day_5": -8.0}, "date": "2026-06-03"},
        {"returns": {"day_5": 2.0}, "date": "2026-06-04"},
    ]

    with patch("backend.db.set_setting") as mock_set:
        result = _summarize_trades(trades, [5], "Gap Up/Down Strategy")
        assert result["untradeable"]
        mock_set.assert_called_with("strategy_status_gap", "untradeable")


@patch("backend.recommender.yf.Ticker")
def test_recommender_filtering(mock_ticker):
    # Mock stock data
    mock_history = MagicMock()
    # Return 60 days of closing prices that are slightly up, volumes, etc.
    # to trigger some signals (e.g. breakout)
    import pandas as pd
    dates = pd.date_range(end="2026-06-07", periods=60)
    # Price rises steadily to trigger breakout
    closes = np.linspace(100, 150, 60)
    highs = closes + 0.1
    lows = closes - 0.1
    opens = closes - 0.05
    volumes = [1000] * 59 + [5000] # volume spike on last day
    
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=dates)
    
    mock_ticker.return_value.history.return_value = df

    # 1. Test when breakout strategy is tradeable
    allowed = {
        "gap": True,
        "volume": True,
        "breakout": True,
        "sr_bounce": True
    }
    
    result = _analyze_stock("RELIANCE", allowed_strategies=allowed)
    assert result is not None
    # Verify breakout signal is generated
    breakout_signals = [s for s in result["signals"] if "Breakout" in s["type"]]
    assert len(breakout_signals) > 0

    # 2. Test when breakout strategy is NOT tradeable (breakout = False)
    blocked = {
        "gap": True,
        "volume": True,
        "breakout": False,
        "sr_bounce": True
    }
    
    result_blocked = _analyze_stock("RELIANCE", allowed_strategies=blocked)
    assert result_blocked is not None
    # Verify breakout signal is NOT generated
    blocked_signals = [s for s in result_blocked["signals"] if "Breakout" in s["type"]]
    assert len(blocked_signals) == 0
