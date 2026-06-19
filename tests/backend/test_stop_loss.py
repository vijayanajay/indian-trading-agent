import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.recommender import compute_atr, _analyze_stock
from backend.honest_assessment import get_honest_assessment, compute_fingerprint
from backend.simulation import open_paper_trade, hit_paper_trade_stop, check_and_trigger_stop_losses
from backend.db import ensure_db

@pytest.fixture(autouse=True)
def setup_db():
    ensure_db()

def test_compute_atr():
    closes = [100.0] * 20
    highs = [105.0] * 20
    lows = [95.0] * 20
    # TR = max(5, 5, 5) = 10
    atr = compute_atr(highs, lows, closes, 14)
    assert atr == 10.0

@patch("backend.recommender._compute_rsi", return_value=50.0)
@patch("backend.recommender.yf.Ticker")
def test_recommender_stop_loss_long(mock_ticker, mock_rsi):
    dates = pd.date_range(end="2026-06-07", periods=60)
    volumes = [1000] * 60
    
    # Setup a stock near support (60d low is 99.0, 60d high is 110.0)
    opens = [100.0] * 59 + [100.5]
    highs = [110.0] * 59 + [100.8]
    lows = [99.0] * 59 + [99.2]
    closes = [100.0] * 59 + [100.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"sr_bounce": True})
    assert result is not None
    assert result["suggested_stop_loss"] is not None
    # Support level is 99.0, which is within 2% fallback (fallback is 2% = 2.0). 
    # Since 100.0 - 99.0 = 1.0 <= 2.0, suggested_stop_loss should be 99.0!
    assert result["suggested_stop_loss"] == 99.0
    assert "Nearest support level" in result["invalidation_reason"]

@patch("backend.recommender._compute_rsi", return_value=50.0)
@patch("backend.recommender.yf.Ticker")
def test_recommender_stop_loss_fallback(mock_ticker, mock_rsi):
    dates = pd.date_range(end="2026-06-07", periods=60)
    volumes = [1000] * 59 + [2000] # Volume spike to trigger confirmed breakout
    
    # 20-day high is 80.0. 60-day high is 80.0. Today is a breakout at 100.0. Support is 80.0 (too far).
    opens = [80.0] * 59 + [99.5]
    highs = [80.0] * 59 + [101.0]
    lows = [80.0] * 59 + [99.5]
    closes = [80.0] * 59 + [100.0]
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)
    mock_ticker.return_value.history.return_value = df
    
    result = _analyze_stock("TEST", allowed_strategies={"breakout": True})
    assert result is not None
    assert result["suggested_stop_loss"] is not None
    # ATR fallback of 100 with ATR of 1.5 is 97.0
    assert result["suggested_stop_loss"] == 97.0
    assert "Max ATR-based fallback" in result["invalidation_reason"]

def test_honest_assessment_with_rr():
    # Test that get_honest_assessment uses risk_reward_ratio as Kelly b
    signals = [{"type": "Volume Spike (Bullish)", "direction": "BULLISH"}]
    fingerprint = compute_fingerprint(["Volume Spike (Bullish)"], "BULL")
    
    # Mock model coefficients and predict_win_probability to force Tier 4 Calibrated
    with patch("backend.signal_model.load_model_coefficients", return_value=({"test": 1.0}, 0.60, 0.15)), \
         patch("backend.signal_model.predict_win_probability", return_value=0.60), \
         patch("backend.honest_assessment.get_db") as mock_db, \
         patch("backend.honest_assessment.get_portfolio_drawdown", return_value=0.0):
         
         # Mock database returns to satisfy Tier 4 Calibrated trade counting (>100 trades)
         # We return 120 wins/losses to ensure n_trades = 120 (>=100)
         mock_conn = MagicMock()
         mock_conn.execute.return_value.fetchone.return_value = {"cnt": 0} # cache empty
         
         rows = []
         # 100 wins
         for i in range(1, 101):
             rows.append({
                 "source": "paper", "ticker": "AAPL", "entry_date": f"2026-01-{i:02d}",
                 "pnl_5d_pct": 5.0, "signal_fingerprint": fingerprint, "triggered_signals": "[]", "regime_at_entry": "BULL"
             })
         # 20 losses
         for i in range(1, 21):
             rows.append({
                 "source": "paper", "ticker": "AAPL", "entry_date": f"2026-02-{i:02d}",
                 "pnl_5d_pct": -2.0, "signal_fingerprint": fingerprint, "triggered_signals": "[]", "regime_at_entry": "BULL"
             })
         mock_conn.execute.return_value.fetchall.return_value = rows
         mock_db.return_value.__enter__.return_value = mock_conn
         
         # Case 1: passing risk_reward_ratio = 3.0
         assessment = get_honest_assessment(signals, 4.0, "BULL", risk_reward_ratio=3.0)
         # p = 0.6, b = 3.0, q = 0.4
         # k_frac = (0.6 * 3 - 0.4) / 3 = 1.4 / 3 = 0.4667 -> 46.7%
         # Capped at 15.0%
         assert assessment["kelly_pct"] == 15.0
         
         # Case 2: passing risk_reward_ratio = 1.0
         assessment2 = get_honest_assessment(signals, 4.0, "BULL", risk_reward_ratio=1.0)
         # p = 0.6, b = 1.0, q = 0.4
         # k_frac = (0.6 * 1 - 0.4) / 1 = 0.2 -> 20.0%
         # Capped at 15.0%
         assert assessment2["kelly_pct"] == 15.0

@patch("backend.simulation.yf.Ticker")
@patch("backend.simulation.add_paper_trade")
def test_open_paper_trade_auto_populate(mock_add_trade, mock_ticker):
    # Setup mock history for entry price
    df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range(end="2026-06-07", periods=1))
    mock_ticker.return_value.history.return_value = df
    
    # We mock _analyze_stock to return suggested_stop_loss = 98.0 and risk_reward_ratio = 2.0
    with patch("backend.recommender._analyze_stock") as mock_analyze:
        mock_analyze.return_value = {
            "suggested_stop_loss": 98.0,
            "risk_reward_ratio": 2.0,
        }
        
        open_paper_trade("AAPL", source="recommendation", signal="BUY")
        
        mock_add_trade.assert_called_once()
        called_arg = mock_add_trade.call_args[0][0]
        assert called_arg["stop_loss_price"] == 98.0
        assert called_arg["risk_reward_ratio"] == 2.0

@patch("backend.simulation.yf.Ticker")
@patch("backend.simulation.add_paper_trade")
def test_open_paper_trade_no_overwrite(mock_add_trade, mock_ticker):
    # Setup mock history for entry price
    df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range(end="2026-06-07", periods=1))
    mock_ticker.return_value.history.return_value = df
    
    # We mock _analyze_stock to return suggested_stop_loss = 98.0 and risk_reward_ratio = 2.0
    with patch("backend.recommender._analyze_stock") as mock_analyze:
        mock_analyze.return_value = {
            "suggested_stop_loss": 98.0,
            "risk_reward_ratio": 2.0,
        }
        
        # Scenario: stop_loss_price is provided (95.0), risk_reward_ratio is None
        open_paper_trade("AAPL", source="recommendation", signal="BUY", stop_loss_price=95.0, risk_reward_ratio=None)
        
        mock_add_trade.assert_called_once()
        called_arg = mock_add_trade.call_args[0][0]
        # Should keep caller's stop_loss_price (95.0) and auto-populate risk_reward_ratio (2.0)
        assert called_arg["stop_loss_price"] == 95.0
        assert called_arg["risk_reward_ratio"] == 2.0


@patch("backend.simulation.list_paper_trades")
@patch("backend.simulation.get_db")
@patch("backend.simulation.refresh_paper_trade_prices")
@patch("backend.simulation.update_paper_trade_status")
def test_hit_paper_trade_stop(mock_status, mock_refresh, mock_db, mock_list):
    mock_list.return_value = [{
        "id": 1, "ticker": "AAPL", "entry_price": 100.0, "stop_loss_price": 98.0,
        "direction": "LONG", "status": "active"
    }]
    
    mock_conn = MagicMock()
    mock_db.return_value.__enter__.return_value = mock_conn
    
    res = hit_paper_trade_stop(1, 98.0)
    assert res["ok"] is True
    assert res["close_price"] == 98.0
    assert res["pnl_pct"] == -2.0
    mock_status.assert_called_once_with(1, "hit_stop")

@patch("backend.simulation.list_paper_trades")
@patch("backend.simulation.yf.Ticker")
@patch("backend.simulation.hit_paper_trade_stop")
def test_check_and_trigger_stop_losses(mock_hit_stop, mock_ticker, mock_list):
    mock_list.return_value = [{
        "id": 1, "ticker": "AAPL", "entry_price": 100.0, "stop_loss_price": 98.0,
        "direction": "LONG", "status": "active"
    }]
    
    # Scenario: Low of the day is 97.0, which breaches stop of 98.0
    df = pd.DataFrame({"Low": [97.0], "High": [101.0], "Close": [99.0]}, index=pd.date_range(end="2026-06-07", periods=1))
    mock_ticker.return_value.history.return_value = df
    
    check_and_trigger_stop_losses()
    mock_hit_stop.assert_called_once_with(1, 98.0)
