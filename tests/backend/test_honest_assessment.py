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

    # Setup valid calibrated model coefficients (Brier = 0.15 < 0.20)
    set_setting("calibration_model_beta_0", "-1.5")
    set_setting("calibration_model_beta_1", "0.5")
    set_setting("calibration_model_brier", "0.15")
    
    # Tier 4: Calibrated (n >= 100 + calibrated model + Brier < 0.20)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["probability"] is not None
    assert assessment["kelly_pct"] is not None
    assert assessment["suggested_position_size_pct"] == assessment["kelly_pct"]
    assert "Model:" in assessment["display_message"]
    
    # Check fallback when Brier is too high (Brier = 0.22 >= 0.20)
    set_setting("calibration_model_brier", "0.22")
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

