"""Unit tests for the Honest Assessment Engine and model calibration pipeline."""

import os
import sqlite3
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Set env before imports
os.environ["DB_PATH"] = ":memory:"

from backend.db import get_db, ensure_db, set_setting, get_setting
from backend.honest_assessment import (
    compute_fingerprint,
    wilson_confidence_interval,
    get_honest_assessment,
)
from backend.cron import fit_logistic_regression, retrain_calibration_model


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create a clean in-memory database before each test."""
    ensure_db()
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        conn.execute("DELETE FROM signal_performance_cache")
        conn.execute("DELETE FROM settings")
    yield
    # Cleanup settings
    set_setting("calibration_model_beta_0", None)
    set_setting("calibration_model_beta_1", None)
    set_setting("calibration_model_brier", None)
    set_setting("calibration_model_trained_at", None)
    set_setting("calibration_model_n_trades", None)


def test_fingerprint_computation():
    """Test that fingerprint hashing is order-invariant and deterministic."""
    signals_a = ["Breakout (Volume Confirmed)", "Strong Uptrend"]
    signals_b = ["Strong Uptrend", "Breakout (Volume Confirmed)"]
    
    fp_a = compute_fingerprint(signals_a, "BULL")
    fp_b = compute_fingerprint(signals_b, "BULL")
    
    assert fp_a == fp_b
    assert len(fp_a) == 64  # SHA256 length in hex representation
    
    # Different regime should yield different hash
    fp_c = compute_fingerprint(signals_a, "BEAR")
    assert fp_a != fp_c


def test_wilson_confidence_interval():
    """Verify Wilson confidence interval limits."""
    # Zero sample size
    low, high = wilson_confidence_interval(0, 0)
    assert low == 0.0
    assert high == 0.0

    # 100% win rate
    low, high = wilson_confidence_interval(10, 10)
    assert low > 0.60
    assert high == 1.0

    # 50% win rate
    low, high = wilson_confidence_interval(50, 100)
    assert abs(low - 0.43) < 0.02
    assert abs(high - 0.57) < 0.02


def test_fit_logistic_regression():
    """Test that the NumPy-based logistic regression solves synthetic data correctly."""
    # Generate linear boundary where higher score leads to success
    np.random.seed(42)
    scores = np.random.uniform(1.0, 5.0, 100)
    # y = 1 if score >= 3, else 0 (with a bit of noise)
    probs = 1.0 / (1.0 + np.exp(-(-3.0 + 1.2 * scores)))
    successes = (np.random.rand(100) < probs).astype(int)

    beta_0, beta_1 = fit_logistic_regression(scores, successes)
    
    # Intercept should be negative, coefficient should be positive (positive relationship)
    assert beta_0 < 0
    assert beta_1 > 0


def test_honest_assessment_tiers():
    """Verify HonestAssessmentEngine transitions across Exploratory, Emerging, Empirical, and Calibrated tiers."""
    with get_db() as conn:
        conn.execute("DELETE FROM model_coefficients")
        import backend.signal_model
        backend.signal_model._MODEL_CACHE = None

    signals = [{"type": "Breakout (Volume Confirmed)"}, {"type": "Strong Uptrend"}]
    score = 4.5
    regime = "BULL"
    
    # Tier 1: Exploratory (n_trades = 0)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "EXPLORATORY"
    assert assessment["suggested_position_size_pct"] == 5.0
    assert assessment["probability"] is None
    
    # Mock database connection to return custom counts for cache/dynamic lookups
    fingerprint = assessment["fingerprint"]
    
    with get_db() as conn:
        # Insert mock cache rows
        conn.execute(
            """INSERT INTO signal_performance_cache (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (fingerprint, 20, 12, 0.60, 0.45, 0.73, 1.5),
        )
        
    # Tier 2: Emerging (10 <= n < 30)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "EMERGING"
    assert assessment["suggested_position_size_pct"] == 7.5
    assert assessment["probability"] is None

    # Update cache to Empirical boundaries
    with get_db() as conn:
        conn.execute(
            "UPDATE signal_performance_cache SET n_trades = 50, wins = 35, win_rate = 0.70, wilson_lower = 0.61, wilson_upper = 0.78 WHERE fingerprint = ?",
            (fingerprint,),
        )
        
    # Tier 3: Empirical (30 <= n < 100)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "EMPIRICAL"
    assert assessment["suggested_position_size_pct"] == 10.0
    assert assessment["probability"] is None
    assert "Historical: 70% win rate" in assessment["display_message"]
    
    # Update cache to Calibrated boundaries
    with get_db() as conn:
        conn.execute(
            "UPDATE signal_performance_cache SET n_trades = 120, wins = 80, win_rate = 0.66, wilson_lower = 0.60, wilson_upper = 0.72 WHERE fingerprint = ?",
            (fingerprint,),
        )
        
    # Calibrated requires trained model in settings. Check fallback to Empirical when model is missing.
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "EMPIRICAL"  # Falls back to Empirical because model coefficients aren't in settings

    # Setup valid calibrated model coefficients
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('intercept', -1.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('sig_breakout_vol_confirmed', 1.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('sig_uptrend_strong', 1.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('regime_BULL', 1.5, 0.60, 0.15, '2026-06-07')")
        # Clear model cache to pick up new coefficients
        import backend.signal_model
        backend.signal_model._MODEL_CACHE = None
    
    # Tier 4: Calibrated (n >= 100 + calibrated model + Brier < 0.25)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["probability"] is not None
    assert assessment["kelly_pct"] is not None
    assert assessment["suggested_position_size_pct"] == assessment["kelly_pct"]
    assert "Model:" in assessment["display_message"]
    
    # Check fallback when Brier is too high (Brier = 0.26 >= 0.25)
    with get_db() as conn:
        conn.execute("UPDATE model_coefficients SET brier = 0.26")
        backend.signal_model._MODEL_CACHE = None
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "EMPIRICAL"  # Bypassed model due to Brier safety check


def test_fingerprint_insertion_and_migration():
    """Verify that fingerprints are computed on paper and shadow trade insertion, backfilled correctly, and fallback queries work."""
    from backend.db import add_paper_trade
    from backend.shadow_trades import record_shadow_trades_from_recommendations
    from backend.backfill_fingerprints import backfill_fingerprints
    import json

    # 1. Test paper trade insert fingerprint computation
    signals = [{"type": "Breakout"}, {"type": "RSI Oversold"}]
    trade_id = add_paper_trade({
        "ticker": "RELIANCE",
        "entry_price": 2500.0,
        "triggered_signals": signals,
        "regime_at_entry": "BULL",
    })

    expected_fp = compute_fingerprint(["Breakout", "RSI Oversold"], "BULL")

    with get_db() as conn:
        row = conn.execute("SELECT signal_fingerprint FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
        assert row is not None
        assert row["signal_fingerprint"] == expected_fp

    # 2. Test shadow trade insert fingerprint computation
    recs = {
        "strong_buys": [{
            "ticker": "TCS",
            "price": 3200.0,
            "confidence": "HIGH",
            "signals": [{"type": "Volume Spike"}],
            "direction": "LONG",
            "score": 4.5,
            "success_probability": 70,
        }]
    }
    
    # We need to mock get_current_regime to return "BULL"
    with patch("backend.market_regime.get_current_regime", return_value={"regime": "BULL"}):
        res = record_shadow_trades_from_recommendations(recs)
        assert res["recorded"] == 1

    expected_shadow_fp = compute_fingerprint(["Volume Spike"], "BULL")
    with get_db() as conn:
        row = conn.execute("SELECT signal_fingerprint FROM shadow_trades WHERE ticker = 'TCS'").fetchone()
        assert row is not None
        assert row["signal_fingerprint"] == expected_shadow_fp

    # 3. Test backfill migration
    # Insert trades with NULL fingerprints
    with get_db() as conn:
        # Paper trade without fingerprint
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, signal_fingerprint)
               VALUES (?, ?, ?, ?, NULL)""",
            ("INFY", 1400.0, json.dumps([{"type": "Gap Up"}]), "BEAR")
        )
        # Shadow trade without fingerprint
        conn.execute(
            """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, NULL)""",
            ("WIPRO", "2026-06-01", 400.0, json.dumps([{"type": "Gap Up"}]), "BEAR")
        )

    # Run backfill
    stats = backfill_fingerprints()
    assert stats["paper_updated"] >= 1
    assert stats["shadow_updated"] >= 1

    expected_backfilled_fp = compute_fingerprint(["Gap Up"], "BEAR")
    with get_db() as conn:
        row_paper = conn.execute("SELECT signal_fingerprint FROM paper_trades WHERE ticker = 'INFY'").fetchone()
        assert row_paper["signal_fingerprint"] == expected_backfilled_fp

        row_shadow = conn.execute("SELECT signal_fingerprint FROM shadow_trades WHERE ticker = 'WIPRO'").fetchone()
        assert row_shadow["signal_fingerprint"] == expected_backfilled_fp

    # 4. Test fallback query in get_honest_assessment (with cache miss)
    # Clear cache and insert historical closed trades under the "Gap Up" (BEAR) fingerprint
    with get_db() as conn:
        conn.execute("DELETE FROM signal_performance_cache")
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        
        # Insert 15 closed trades (e.g. 10 paper, 5 shadow) to trigger EMERGING tier (10 <= n < 30)
        # All with pnl_5d_pct set (so they count as closed)
        for i in range(10):
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"P_{i}", 100.0, json.dumps([{"type": "Gap Up"}]), "BEAR", 1.5 if i % 2 == 0 else -1.0, expected_backfilled_fp)
            )
        for i in range(5):
            conn.execute(
                """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"S_{i}", f"2026-06-0{i}", 100.0, json.dumps([{"type": "Gap Up"}]), "BEAR", 2.0, expected_backfilled_fp)
            )

    # Call get_honest_assessment for "Gap Up" under BEAR regime
    assessment = get_honest_assessment([{"type": "Gap Up"}], 3.0, "BEAR")
    assert assessment["n_trades"] == 15
    assert assessment["tier"] == "EMERGING"
    assert assessment["suggested_position_size_pct"] == 7.5


def test_kelly_criterion_replacement():
    """Verify payoff ratio calculation, caps, drawdown limits, and negative overrides."""
    signals = [{"type": "Breakout (Volume Confirmed)"}, {"type": "Strong Uptrend"}]
    score = 4.5
    regime = "BULL"
    fp = compute_fingerprint([s["type"] for s in signals], regime)
    
    # 1. Setup a valid calibrated model in model_coefficients table
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('intercept', -1.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('sig_breakout_vol_confirmed', 1.0, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('sig_uptrend_strong', 0.75, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('regime_BULL', 0.5, 0.60, 0.15, '2026-06-07')")
        import backend.signal_model
        backend.signal_model._MODEL_CACHE = None
    
    # Update cache to indicate CALIBRATED tier (n >= 100)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO signal_performance_cache (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
               VALUES (?, 120, 80, 0.66, 0.60, 0.72, 1.5)""",
            (fp,),
        )
        
    # Clear paper_trades and shadow_trades first
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        
    # Test 1.1: Default payoff ratio b = 1.0 (no trades in DB yet)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["low_confidence"] is True
    # logit = -1.5 + 1.0 + 0.75 + 0.5 = 0.75 => p = 0.679. b = 1.0. k_frac = 0.679 - 0.321 = 0.358 => 35.8% capped at 15%
    assert assessment["kelly_pct"] == 15.0
    assert "(low confidence)" in assessment["display_message"]
    
    # Test 1.2: Insert trades with average win 4.0% and average loss -2.0%
    # b = 4.0 / 2.0 = 2.0.
    # logit = 0.75 => p = 0.679. q = 0.321.
    # k_frac = (p*b - q)/b = (0.679 * 2.0 - 0.321) / 2.0 = (1.358 - 0.321)/2.0 = 1.037/2.0 = 0.5185 => 51.9% capped at 15.0%
    with get_db() as conn:
        # 1 win and 1 loss
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("T1", 100.0, "[]", "BULL", 4.0, fp)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("T2", 100.0, "[]", "BULL", -2.0, fp)
        )
        
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["low_confidence"] is False
    assert assessment["kelly_pct"] == 15.0
    assert "(low confidence)" not in assessment["display_message"]
    
    # Test 1.3: Capping check with lower p to see actual uncapped Kelly sizing.
    # We update the intercept coefficient to simulate logit = -0.5
    # logit = -2.75 + 1.0 + 0.75 + 0.5 = -0.5 => p = 1/(1 + e^0.5) = 0.3775 => q = 0.6225
    # Since win rate p = 37.75% and b = 2.0, edge is positive: p*b - q = 0.3775*2 - 0.6225 = 0.755 - 0.6225 = 0.1325
    # k_frac = 0.1325 / 2 = 0.06625 => kelly_pct = 6.6%
    with get_db() as conn:
        conn.execute("UPDATE model_coefficients SET coefficient = -2.75 WHERE feature = 'intercept'")
        backend.signal_model._MODEL_CACHE = None
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["kelly_pct"] == 6.6
    assert assessment["suggested_position_size_pct"] == 6.6
    
    # Test 1.4: Negative Kelly override.
    # We update the intercept coefficient to simulate logit = -1.0
    # logit = -3.25 + 1.0 + 0.75 + 0.5 = -1.0 => p = 1/(1+e) = 0.2689 => q = 0.7311
    # p*b - q = 0.2689 * 2 - 0.7311 = 0.5378 - 0.7311 < 0 (negative edge)
    # Kelly should be overridden to DO NOT TRADE / 0%
    with get_db() as conn:
        conn.execute("UPDATE model_coefficients SET coefficient = -3.25 WHERE feature = 'intercept'")
        backend.signal_model._MODEL_CACHE = None
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["kelly_pct"] == 0.0
    assert assessment["suggested_position_size_pct"] == 0.0
    assert assessment["display_message"] == "DO NOT TRADE"

    # Test 1.5: Drawdown override.
    # Reset intercept back to a positive edge level (-1.5)
    with get_db() as conn:
        conn.execute("UPDATE model_coefficients SET coefficient = -1.5 WHERE feature = 'intercept'")
        backend.signal_model._MODEL_CACHE = None

    # We simulate a portfolio down more than 10% from its peak equity.
    # T1 alloc = 10,000. PnL -90% => returned 1,000. Equity = 91,000. Drawdown = 9% from peak 100,000.
    # T2 alloc = 9,100. PnL -50% => returned 4,550. Equity = 81,900 + 4,550 = 86,450. Drawdown = 13.55%.
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("DT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", -90.0, "2026-06-02 10:00:00")
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("DT2", 100.0, "[]", "2026-06-03 10:00:00", "expired", -50.0, "2026-06-04 10:00:00")
        )
        
    assessment = get_honest_assessment(signals, 4.5, regime)
    assert assessment["kelly_pct"] == 0.0
    assert assessment["suggested_position_size_pct"] == 0.0
    assert "DO NOT TRADE (portfolio drawdown > 10%)" in assessment["display_message"]


def test_get_portfolio_drawdown_calculation():
    """Directly test get_portfolio_drawdown with different trade outcomes and timing."""
    from backend.honest_assessment import get_portfolio_drawdown

    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")

    # Scenario 1: Empty database should return 0.0 drawdown
    assert get_portfolio_drawdown() == 0.0

    # Scenario 2: Single win (equity rises, no drawdown)
    # T1 alloc = 10% of 100k = 10k. Returns +20% => returns 12k. Equity = 102k. Drawdown = 0.0.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("WT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", 20.0, "2026-06-02 10:00:00")
        )
    assert get_portfolio_drawdown() == 0.0

    # Scenario 3: Add a loss that causes drawdown from the new peak
    # Equity is 102k (peak).
    # T2 alloc = 10% of 102k = 10.2k. Returns -50% => returns 5.1k.
    # Total equity = 91.8k cash + 5.1k returned = 96.9k.
    # Drawdown from peak (102k) is (102 - 96.9) / 102 * 100 = 5.0%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("WT2", 100.0, "[]", "2026-06-03 10:00:00", "expired", -50.0, "2026-06-04 10:00:00")
        )
    dd = get_portfolio_drawdown()
    assert abs(dd - 5.0) < 0.01



