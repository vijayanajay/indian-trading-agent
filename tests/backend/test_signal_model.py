"""Unit tests for the L1-Regularized Signal Model."""

import os
import json
import sqlite3
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Set env before imports
os.environ["DB_PATH"] = ":memory:"

from backend.db import get_db, ensure_db, set_setting, get_setting
from backend.signal_model import (
    extract_features,
    sigmoid,
    soft_threshold,
    fit_l1_logistic_regression,
    compute_auc,
    compute_brier,
    load_model_coefficients,
    predict_win_probability,
    train_signal_model,
    check_and_trigger_retraining,
    count_closed_trades,
    FEATURE_NAMES,
    SIGNAL_KEYS,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create a clean in-memory database before each test."""
    ensure_db()
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        conn.execute("DELETE FROM model_coefficients")
        conn.execute("DELETE FROM settings")
    yield


def test_extract_features():
    """Verify that features are extracted correctly and length is 28."""
    # Active signals
    signals = [{"type": "Volume Spike (Bullish)"}, {"type": "Breakout (Volume Confirmed)"}]
    regime = "BULL"
    
    feats = extract_features(signals, regime)
    
    # Check length: intercept (1) + signals (17) + regimes (4) + interactions (6) = 28
    assert len(feats) == 28
    
    # Check intercept
    assert feats["intercept"] == 1.0
    
    # Check active signals
    assert feats["sig_volume_bullish"] == 1.0
    assert feats["sig_breakout_vol_confirmed"] == 1.0
    assert feats["sig_rsi_oversold"] == 0.0
    
    # Check regime
    assert feats["regime_BULL"] == 1.0
    assert feats["regime_BEAR"] == 0.0
    
    # Check interaction terms
    # breakout_vol_confirmed x volume_bullish should be active
    assert feats["int_breakout_vol_confirmed_x_volume_bullish"] == 1.0
    # breakout_weak x volume_bullish should not be active
    assert feats["int_breakout_weak_x_volume_bullish"] == 0.0


def test_sigmoid():
    """Test sigmoid boundary conditions."""
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) > 0.99
    assert sigmoid(-100.0) < 0.01


def test_soft_threshold():
    """Test soft thresholding operator."""
    assert soft_threshold(2.0, 0.5) == 1.5
    assert soft_threshold(-2.0, 0.5) == -1.5
    assert soft_threshold(0.3, 0.5) == 0.0
    assert soft_threshold(-0.3, 0.5) == 0.0


def test_fit_l1_logistic_regression():
    """Test the Proximal Gradient Descent solver with a simple dataset."""
    np.random.seed(42)
    # Generate simple separable data
    X = np.random.randn(100, 5)
    X[:, 0] = 1.0  # intercept
    # Define true weights where only index 1 and 2 are active (sparse)
    w_true = np.array([0.5, 2.0, -1.5, 0.0, 0.0])
    
    y_prob = sigmoid(np.dot(X, w_true))
    y = (np.random.rand(100) < y_prob).astype(float)
    
    # Fit with L1 penalty
    w_fit = fit_l1_logistic_regression(X, y, alpha=0.01, lr=0.1, max_iter=2000)
    
    assert len(w_fit) == 5
    # The sparse coefficients should be small or 0
    assert abs(w_fit[3]) < 0.2
    assert abs(w_fit[4]) < 0.2


def test_metrics_calculation():
    """Test AUC and Brier score calculators."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    
    auc = compute_auc(y_true, y_prob)
    assert auc == 1.0
    
    brier = compute_brier(y_true, y_prob)
    # Mean of (0.1^2 + 0.1^2 + 0.2^2 + 0.2^2) = (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.10 / 4 = 0.025
    assert abs(brier - 0.025) < 1e-6


def test_model_training_and_inference():
    """Verify that model training correctly fits weights, runs validation, and promotes the model."""
    # 1. Cold state fallback verification
    # Try training with 0 trades - should warn and fall back
    res = train_signal_model()
    assert res["status"] == "warning"
    assert "Insufficient data" in res["message"]
    
    # Verify inference returns None when no model exists
    prob = predict_win_probability("dummy_fp", "BULL")
    assert prob is None
    
    # 2. Insert closed mock trades to exceed the 50 trade threshold
    with get_db() as conn:
        for i in range(60):
            # Alternate wins/losses to build clean outcome matrix
            pnl = 2.0 if i % 2 == 0 else -1.5
            
            # Simple signals representation:
            # For even i, we trigger 'Volume Spike (Bullish)' and 'Breakout (Volume Confirmed)' -> BULLISH outcome
            # For odd i, we trigger 'Volume Spike (Bearish)' and 'Breakdown Below Support' -> BEARISH outcome
            if i % 2 == 0:
                signals_json = json.dumps([{"type": "Volume Spike (Bullish)"}, {"type": "Breakout (Volume Confirmed)"}])
                regime = "BULL"
            else:
                signals_json = json.dumps([{"type": "Volume Spike (Bearish)"}, {"type": "Breakdown Below Support"}])
                regime = "BEAR"
                
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint, entry_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"TC_{i}", 100.0, signals_json, regime, pnl, f"fp_{i}", f"2026-06-0{i%9 + 1}"),
            )
            
    # Count closed trades
    cnt = count_closed_trades()
    assert cnt == 60
    
    # 3. Train model - should succeed and satisfy CV criteria
    res = train_signal_model()
    assert res["status"] == "success"
    assert res["auc"] > 0.55
    assert res["brier"] < 0.25
    
    # Verify coefficients exist in DB
    coefs, auc, brier = load_model_coefficients()
    assert len(coefs) == len(FEATURE_NAMES)
    assert auc == res["auc"]
    assert brier == res["brier"]
    
    # 4. Verify inference works
    test_signals = [{"type": "Volume Spike (Bullish)"}, {"type": "Breakout (Volume Confirmed)"}]
    prob = predict_win_probability("dummy_fp", "BULL", signals=test_signals)
    assert prob is not None
    assert 0.0 <= prob <= 1.0
    
    # Since these signals are correlated with P&L > 0 (even indices), probability should be high
    assert prob > 0.50
    
    # Test signals associated with negative outcomes (odd indices)
    bearish_signals = [{"type": "Volume Spike (Bearish)"}, {"type": "Breakdown Below Support"}]
    prob_bear = predict_win_probability("dummy_fp", "BEAR", signals=bearish_signals)
    assert prob_bear < 0.50


def test_retraining_trigger():
    """Verify check_and_trigger_retraining triggers background retrains when threshold exceeds 20."""
    set_setting("model_last_trained_trade_count", "30")
    
    # Insert closed trades so total = 52
    with get_db() as conn:
        for i in range(52):
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, pnl_5d_pct, triggered_signals, entry_date)
                   VALUES (?, 100.0, 1.0, '[]', ?)""",
                (f"T_{i}", f"2026-06-0{i%9+1}"),
            )
            
    # Total count = 52. Last trained at 30. Diff = 22 >= 20.
    # We mock train_signal_model to verify it gets called
    with patch("backend.signal_model.train_signal_model") as mock_train:
        check_and_trigger_retraining()
        
        # Give thread a split second to launch
        import time
        time.sleep(0.1)
        mock_train.assert_called_once()


def test_deterministic_deduplication():
    """Verify that paper trades are preferred over shadow trades on duplicate ticker/entry_date."""
    # Insert 50 baseline trades to satisfy threshold
    with get_db() as conn:
        for i in range(50):
            pnl = 2.0 if i % 2 == 0 else -1.5
            signals_json = json.dumps([{"type": "Volume Spike (Bullish)"}])
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint, entry_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"TC_{i}", 100.0, signals_json, "BULL", pnl, f"fp_{i}", "2026-06-01"),
            )
        
        # Now insert a duplicate shadow trade for TC_0 on 2026-06-01
        # It has a massive negative P&L. If this shadow trade were selected, the outcome would change.
        conn.execute(
            """INSERT INTO shadow_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint, signal_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("TC_0", 100.0, json.dumps([{"type": "Volume Spike (Bearish)"}]), "BEAR", -100.0, "fp_shadow", "2026-06-01"),
        )
    
    # We patch fit_l1_logistic_regression to inspect the inputs X and y that train_signal_model prepares
    with patch("backend.signal_model.fit_l1_logistic_regression") as mock_fit:
        mock_fit.return_value = np.zeros(28)
        
        res = train_signal_model()
        
        # We expect 50 unique samples (meaning the duplicate shadow trade was discarded)
        assert res["n_samples"] == 50
        
        # Inspect the X and y passed to fit_l1_logistic_regression
        assert mock_fit.called
        args, kwargs = mock_fit.call_args
        X_passed, y_passed = args[0], args[1]
        
        # regime_BEAR should not be present in any sample, because the duplicate shadow trade is BEAR and should have been discarded
        regime_bear_idx = FEATURE_NAMES.index("regime_BEAR")
        for features in X_passed:
            assert features[regime_bear_idx] == 0.0, "Duplicate shadow trade with BEAR regime was not discarded!"


def test_retraining_concurrency_lock():
    """Verify that train_signal_model acquires _RETRAIN_LOCK and rejects concurrent execution."""
    import time
    import threading
    
    # We patch _train_signal_model_internal to sleep for 0.5s so we have time to call concurrently
    def mock_internal():
        time.sleep(0.5)
        return {"status": "success"}
        
    with patch("backend.signal_model._train_signal_model_internal", side_effect=mock_internal):
        results = []
        
        def run_train():
            res = train_signal_model()
            results.append(res)
            
        t1 = threading.Thread(target=run_train)
        t2 = threading.Thread(target=run_train)
        
        t1.start()
        time.sleep(0.1)  # Ensure t1 starts and acquires the lock
        t2.start()
        
        t1.join()
        t2.join()
        
        # One run should succeed, the other should warning-skip because lock is held
        statuses = [r["status"] for r in results]
        assert "success" in statuses
        assert "warning" in statuses
        
        # The skipped one should have the exact warning message
        warning_res = next(r for r in results if r["status"] == "warning")
        assert "already in progress" in warning_res["message"]

