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
    assert assessment["display_message"] == "Paper trade only — no probability estimate"
    
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
    assert assessment["display_message"] == "Building track record — no probability estimate"

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
    assert assessment["display_message"] == "Historical win rate: 70% (61%-78% confidence)"
    
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
    
    # Tier 4: Calibrated (n >= 100 + calibrated model + Brier < 0.20)
    # Test 4.1: Low confidence fallback (no win/loss trades yet)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["probability"] is not None
    assert assessment["low_confidence"] is True
    assert assessment["kelly_pct"] == 0.0
    assert assessment["suggested_position_size_pct"] == 0.0
    assert assessment["display_message"] == "DO NOT TRADE (insufficient win/loss data for Kelly sizing)"

    # Insert a win and a loss to establish high confidence
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("T1", 100.0, "[]", "BULL", 2.0, fingerprint)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("T2", 100.0, "[]", "BULL", -1.0, fingerprint)
        )

    # Test 4.2: High confidence Calibrated
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["probability"] is not None
    assert assessment["low_confidence"] is False
    assert assessment["kelly_pct"] > 0.0
    assert assessment["suggested_position_size_pct"] == assessment["kelly_pct"]
    assert "Model:" in assessment["display_message"]
    
    # Check fallback when Brier is too high (Brier = 0.21 >= 0.20)
    with get_db() as conn:
        conn.execute("UPDATE model_coefficients SET brier = 0.21")
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
        
    # Test 1.1: Default payoff ratio (no trades in DB yet -> low confidence -> Kelly sizing forced to 0.0)
    assessment = get_honest_assessment(signals, score, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["low_confidence"] is True
    assert assessment["kelly_pct"] == 0.0
    assert assessment["suggested_position_size_pct"] == 0.0
    assert assessment["display_message"] == "DO NOT TRADE (insufficient win/loss data for Kelly sizing)"
    
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
        conn.execute("DELETE FROM shadow_trades")
        # Insert 1 win and 1 loss under shadow_trades so low_confidence is False
        conn.execute(
            """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("ST1", "2026-06-01", 100.0, "[]", "BULL", 4.0, fp)
        )
        conn.execute(
            """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("ST2", "2026-06-02", 100.0, "[]", "BULL", -2.0, fp)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("DT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", -90.0, "2026-06-02 10:00:00", 10.0)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("DT2", 100.0, "[]", "2026-06-03 10:00:00", "expired", -50.0, "2026-06-04 10:00:00", 10.0)
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
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", 20.0, "2026-06-02 10:00:00", 10.0)
        )
    assert get_portfolio_drawdown() == 0.0

    # Scenario 3: Add a loss that causes drawdown from the new peak
    # Equity is 102k (peak).
    # T2 alloc = 10% of 102k = 10.2k. Returns -50% => returns 5.1k.
    # Total equity = 91.8k cash + 5.1k returned = 96.9k.
    # Drawdown from peak (102k) is (102 - 96.9) / 102 * 100 = 5.0%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT2", 100.0, "[]", "2026-06-03 10:00:00", "expired", -50.0, "2026-06-04 10:00:00", 10.0)
        )
    dd = get_portfolio_drawdown()
    assert abs(dd - 5.0) < 0.01

    # Scenario 4: NULL position_size_pct fallback to 5.0%
    # Cash before WT3 entry: 91.8k + 5.1k = 96.9k.
    # WT3 alloc = 96.9k * 5.0% = 4.845k (using fallback).
    # Cash after entry: 92.055k.
    # WT3 returns -50.0% => returned = 2.4225k.
    # Current equity = 92.055k + 2.4225k = 94.4775k.
    # Drawdown from peak (102k) is (102 - 94.4775) / 102 * 100 = 7.375%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
            ("WT3", 100.0, "[]", "2026-06-05 10:00:00", "expired", -50.0, "2026-06-06 10:00:00")
        )
    dd_fallback = get_portfolio_drawdown()
    assert abs(dd_fallback - 7.375) < 0.01


def test_position_size_migration():
    """Verify that _run_position_size_migration correctly backfills position_size_pct."""
    from backend.db import _run_position_size_migration
    
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        # Insert 4 trades with NULL position_size_pct
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, success_probability, position_size_pct)
               VALUES (?, ?, ?, NULL)""",
            ("M1", 100.0, 70)  # calibrated >= 65 -> 10.0%
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, success_probability, position_size_pct)
               VALUES (?, ?, ?, NULL)""",
            ("M2", 100.0, 60)  # calibrated >= 55 -> 7.5%
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, success_probability, position_size_pct)
               VALUES (?, ?, ?, NULL)""",
            ("M3", 100.0, 40)  # calibrated < 55 -> 5.0%
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, success_probability, position_size_pct)
               VALUES (?, ?, NULL, NULL)""",
            ("M4", 100.0)      # success_probability is NULL -> 5.0%
        )
        # One trade with already set position_size_pct that should NOT be mutated
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, success_probability, position_size_pct)
               VALUES (?, ?, ?, ?)""",
            ("M5", 100.0, 70, 15.0)
        )
        
    _run_position_size_migration()
    
    with get_db() as conn:
        rows = conn.execute("SELECT ticker, position_size_pct FROM paper_trades ORDER BY ticker").fetchall()
        results = {r["ticker"]: r["position_size_pct"] for r in rows}
        
    assert results["M1"] == 10.0
    assert results["M2"] == 7.5
    assert results["M3"] == 5.0
    assert results["M4"] == 5.0
    assert results["M5"] == 15.0


def test_get_honest_assessment_empty_cache_populated_trades():
    """Verify that when cache is empty but trades exist (possibly with NULL fingerprints),
    get_honest_assessment primes/backfills the cache synchronously, and resolves correctly.
    """
    import json
    
    # 1. Clear everything
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        conn.execute("DELETE FROM signal_performance_cache")
    
    signals = [{"type": "Breakout (Volume Confirmed)"}, {"type": "Strong Uptrend"}]
    regime = "BULL"
    expected_fp = compute_fingerprint(["Breakout (Volume Confirmed)", "Strong Uptrend"], regime)
    
    # 2. Insert 15 trades with NULL signal_fingerprint
    with get_db() as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, NULL)""",
                (f"P_{i}", 100.0, json.dumps(signals), regime, 1.5 if i % 2 == 0 else -1.0)
            )
        for i in range(5):
            conn.execute(
                """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (f"S_{i}", "2026-06-01", 100.0, json.dumps(signals), regime, 2.0)
            )
            
    # Verify cache is empty before calling get_honest_assessment
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM signal_performance_cache").fetchone()
        assert row["cnt"] == 0
        
    # 3. Call get_honest_assessment
    # This should trigger synchronous backfill, populating the cache and fingerprints
    assessment = get_honest_assessment(signals, 4.5, regime)
    
    # Verify cache was indeed populated
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM signal_performance_cache").fetchone()
        assert row["cnt"] > 0
        
        # Verify the fingerprints in the database were backfilled
        paper_fps = conn.execute("SELECT DISTINCT signal_fingerprint FROM paper_trades").fetchall()
        assert len(paper_fps) == 1
        assert paper_fps[0]["signal_fingerprint"] == expected_fp
        
        shadow_fps = conn.execute("SELECT DISTINCT signal_fingerprint FROM shadow_trades").fetchall()
        assert len(shadow_fps) == 1
        assert shadow_fps[0]["signal_fingerprint"] == expected_fp
        
    # Verify assessment details (15 trades -> EMERGING tier)
    assert assessment["n_trades"] == 15
    assert assessment["tier"] == "EMERGING"
    assert assessment["suggested_position_size_pct"] == 7.5


def test_fallback_query_resolves_null_fingerprints():
    """Verify that if the cache is populated (not empty), but has no entry for our fingerprint,
    the fallback query successfully resolves trades with NULL fingerprints in Python.
    """
    import json
    
    # 1. Clear everything
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        conn.execute("DELETE FROM signal_performance_cache")
        
        # Populate cache with a dummy entry so cache count > 0 (backfill won't run)
        conn.execute(
            """INSERT INTO signal_performance_cache (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
               VALUES ('dummy_fp', 5, 3, 0.6, 0.3, 0.8, 1.0)"""
        )
        
    signals = [{"type": "Special Signal"}]
    regime = "BEAR"
    expected_fp = compute_fingerprint(["Special Signal"], regime)
    
    # 2. Insert 12 trades with NULL fingerprints matching the signals
    with get_db() as conn:
        for i in range(8):
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, NULL)""",
                (f"P_S_{i}", 100.0, json.dumps(signals), regime, 1.5 if i % 2 == 0 else -1.0)
            )
        for i in range(4):
            conn.execute(
                """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (f"S_S_{i}", "2026-06-01", 100.0, json.dumps(signals), regime, 2.0)
            )
            
    # Call get_honest_assessment. It should hit fallback query (since no cache entry for expected_fp)
    # and find all 12 trades with NULL fingerprints by reconstructing them in Python.
    assessment = get_honest_assessment(signals, 3.5, regime)
    
    assert assessment["n_trades"] == 12
    assert assessment["tier"] == "EMERGING"
    assert assessment["suggested_position_size_pct"] == 7.5


def test_honest_assessment_fallback_and_kelly_deduplication():
    """Verify that paper and shadow trades with the same ticker and date are deduplicated,
    preferring paper over shadow, in both fallback and Kelly stats calculation paths.
    """
    import json
    
    # 1. Clear everything
    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        conn.execute("DELETE FROM signal_performance_cache")
        # Ensure model cache is cleared
        import backend.signal_model
        backend.signal_model._MODEL_CACHE = None

    signals = [{"type": "Special Signal"}]
    regime = "BEAR"
    expected_fp = compute_fingerprint(["Special Signal"], regime)
    
    # Populate cache with a dummy entry so cache count > 0 (synchronous backfill won't run, forcing fallback path)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO signal_performance_cache (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
               VALUES ('dummy_fp', 5, 3, 0.6, 0.3, 0.8, 1.0)"""
        )

    # 2. Insert 10 unique trades, but with duplicates in shadow trades
    # We want to insert 10 unique ticker+date combinations to land exactly in EMERGING tier (10 <= n < 30).
    # If deduplication works, n_trades will be 10. If not, it will be 15.
    with get_db() as conn:
        # 10 paper trades (unique dates)
        for i in range(10):
            conn.execute(
                """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_date, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"REL", 100.0, json.dumps(signals), f"2026-06-{i:02d}", regime, 2.0, expected_fp)
            )
            
        # 5 duplicate shadow trades (same ticker and dates as the first 5 paper trades)
        # These shadow trades have different pnl_5d_pct (e.g. -5.0) to verify they are not chosen.
        for i in range(5):
            conn.execute(
                """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"REL", f"2026-06-{i:02d}", 100.0, json.dumps(signals), regime, -5.0, expected_fp)
            )

    # Call get_honest_assessment.
    # It should:
    # 1. Hit fallback query (since cache has no entry for expected_fp).
    # 2. Find 10 paper + 5 shadow trades, but deduplicate the 5 shadow trades.
    # 3. n_trades should be 10, all wins (pnl = 2.0 > 0), so win_rate = 1.0.
    # (If deduplication failed, n_trades would be 15, and 5 of them would have pnl = -5.0, resulting in a win rate < 1.0).
    assessment = get_honest_assessment(signals, 3.5, regime)
    
    assert assessment["n_trades"] == 10
    assert assessment["tier"] == "EMERGING"
    assert assessment["win_rate"] == 1.0

    # 3. Test Kelly stats block deduplication (within CALIBRATED tier)
    # We need to insert a trained model so CALIBRATED tier can trigger.
    # And we update cache for expected_fp to n_trades = 120 (>= 100) so it qualifies as CALIBRATED.
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('intercept', 0.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('sig_special_signal', 0.5, 0.60, 0.15, '2026-06-07')")
        conn.execute("INSERT OR REPLACE INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date) VALUES ('regime_BEAR', 0.5, 0.60, 0.15, '2026-06-07')")
        
        conn.execute(
            """INSERT OR REPLACE INTO signal_performance_cache (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
               VALUES (?, 120, 80, 0.66, 0.60, 0.72, 1.5)""",
            (expected_fp,),
        )
        
        # Clear paper & shadow trades and insert fresh ones to check Kelly stats block (payoff ratio)
        conn.execute("DELETE FROM paper_trades")
        conn.execute("DELETE FROM shadow_trades")
        
        # Insert 1 paper win, 1 paper loss
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_date, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("REL", 100.0, "[]", "2026-06-01", regime, 4.0, expected_fp)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_date, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("REL", 100.0, "[]", "2026-06-02", regime, -2.0, expected_fp)
        )
        
        # Insert duplicate shadow trades for both, but with huge positive/negative values
        # If deduplication works, these huge values will be ignored and we'll get avg_win=4.0 and avg_loss=-2.0.
        # If deduplication fails, payoff ratio will be distorted.
        conn.execute(
            """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("REL", "2026-06-01", 100.0, "[]", regime, 400.0, expected_fp)
        )
        conn.execute(
            """INSERT INTO shadow_trades (ticker, signal_date, entry_price, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("REL", "2026-06-02", 100.0, "[]", regime, -200.0, expected_fp)
        )
        
        # Clear model coefficients cache to pick up new ones
        import backend.signal_model
        backend.signal_model._MODEL_CACHE = None

    # Call get_honest_assessment
    assessment = get_honest_assessment(signals, 4.5, regime)
    assert assessment["tier"] == "CALIBRATED"
    assert assessment["low_confidence"] is False
    # If deduplication was successful:
    # avg_win = 4.0, avg_loss = -2.0 => payoff ratio b = 4.0 / 2.0 = 2.0
    # logit = 0.5 (intercept) + 0.5 (sig) + 0.5 (regime) = 1.5 => p = 1 / (1 + e^-1.5) = 0.8176 => q = 0.1824
    # kelly_frac = (p*b - q)/b = (0.8176*2.0 - 0.1824)/2.0 = 0.7264 => kelly_pct = 72.6% capped at 15.0%
    assert assessment["kelly_pct"] == 15.0


def test_portfolio_drawdown_with_unrealized_pnl():
    """Verify that portfolio drawdown calculation incorporates unrealized P&L of open positions."""
    from backend.honest_assessment import get_portfolio_drawdown

    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")

    # WT1 (closed win): alloc 10% of 100k = 10k. P&L +20% => returned 12k. Equity = 102k.
    # WT2 (active loss): position_size_pct = 10%. unrealized_pnl_pct = -30.0.
    # Current Equity before WT2 entry: 102k.
    # WT2 entry alloc: 102k * 10% = 10.2k.
    # Cash remaining: 102k - 10.2k = 91.8k.
    # WT2 marked-to-market value: 10.2k * (1 - 0.3) = 7.14k.
    # Total marked-to-market equity: 91.8k + 7.14k = 98.94k.
    # Peak equity: 102k.
    # Drawdown: (102k - 98.94k) / 102k * 100 = 3.0%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", 20.0, "2026-06-02 10:00:00", 10.0, 0.0)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT2", 100.0, "[]", "2026-06-03 10:00:00", "active", None, None, 10.0, -30.0)
        )

    dd = get_portfolio_drawdown()
    assert abs(dd - 3.0) < 0.01


def test_portfolio_drawdown_same_day_trade():
    """Verify that same-day/timestamp entry and exit are processed entry-first so drawdown calculation is correct."""
    from backend.honest_assessment import get_portfolio_drawdown

    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")

    # Trade 1: Same day trade (entry and exit at same timestamp).
    # Entry: 2026-06-01 10:00:00, Exit: 2026-06-01 10:00:00
    # Returns -50.0%, position_size_pct = 10.0%.
    # Trade 2: Next day trade.
    # Entry: 2026-06-02 10:00:00, Exit: 2026-06-03 10:00:00
    # Returns -50.0%, position_size_pct = 10.0%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("ST1", 100.0, "[]", "2026-06-01 10:00:00", "manually_closed", -50.0, "2026-06-01 10:00:00", 10.0, 0.0)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("ST2", 100.0, "[]", "2026-06-02 10:00:00", "expired", -50.0, "2026-06-03 10:00:00", 10.0, 0.0)
        )

    dd = get_portfolio_drawdown()
    # If same-day trades are processed correctly (entry first, then exit):
    # - Start equity: 100k
    # - ST1 entry: alloc 10k (10% of 100k). Cash = 90k.
    # - ST1 exit: popped, returned = 5k. Cash = 95k. Equity = 95k.
    # - ST2 entry: alloc 9.5k (10% of 95k). Cash = 85.5k.
    # - ST2 exit: popped, returned = 4.75k. Cash = 90.25k. Equity = 90.25k.
    # - Drawdown from peak (100k) should be 9.75%
    # If the bug is active (exit first, then entry):
    # - ST1 exit processed first: no-op (ST1 not in open_positions).
    # - ST1 entry: alloc 10k. Cash = 90k, open_positions = {0: 10k}.
    # - ST2 entry: alloc 10k (10% of 100k equity, since open_positions[0] = 10k). Cash = 80k. open_positions = {0: 10k, 1: 10k}.
    # - ST2 exit: popped ST2. returned = 5k. Cash = 85k. open_positions = {0: 10k}.
    # - Final equity = 85k + 10k (marked-to-market ST1) = 95k. Drawdown is 5.0%.
    assert abs(dd - 9.75) < 0.01


def test_portfolio_drawdown_null_pnl():
    """Verify that trades with NULL P&L are skipped from the drawdown simulation."""
    from backend.honest_assessment import get_portfolio_drawdown

    with get_db() as conn:
        conn.execute("DELETE FROM paper_trades")

    # Trade 1: Valid loss. Entry 2026-06-01, Exit 2026-06-02.
    # Returns -50.0%, position_size_pct = 10.0%.
    # Trade 2: Trade with NULL/None P&L (expired with price-fetching failure).
    # Entry 2026-06-03, Exit 2026-06-07.
    # Trade 3: Valid loss. Entry 2026-06-05, Exit 2026-06-06.
    # Returns -50.0%, position_size_pct = 10.0%.
    with get_db() as conn:
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT1", 100.0, "[]", "2026-06-01 10:00:00", "expired", -50.0, "2026-06-02 10:00:00", 10.0, 0.0)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT2", 100.0, "[]", "2026-06-03 10:00:00", "expired", None, "2026-06-07 10:00:00", 10.0, None)
        )
        conn.execute(
            """INSERT INTO paper_trades (ticker, entry_price, triggered_signals, entry_datetime, status, pnl_5d_pct, updated_at, position_size_pct, unrealized_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("WT3", 100.0, "[]", "2026-06-05 10:00:00", "expired", -50.0, "2026-06-06 10:00:00", 10.0, 0.0)
        )

    dd = get_portfolio_drawdown()
    # If WT2 is skipped:
    # WT1 alloc 10% of 100k = 10k. returned = 5k. Cash = 95k. Equity = 95k. Drawdown = 5.0%.
    # WT3 alloc 10% of 95k = 9.5k. returned = 4.75k. Cash = 90.25k. Equity = 90.25k. Drawdown = 9.75%.
    # Max drawdown = 9.75%.
    # If WT2 is not skipped (defaults to 0.0%):
    # WT2 exit at 2026-06-07 returns 9.5k. Final equity goes back to 95.0k. Drawdown at the end is 5.0%.
    assert abs(dd - 9.75) < 0.01








