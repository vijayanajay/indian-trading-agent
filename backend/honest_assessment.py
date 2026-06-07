"""Honest Assessment Engine — replaces fake success probability with a 4-tier data quality classification.

Tiers:
1. EXPLORATORY (n < 10): Insufficient data. Show 50% position size, no probability.
2. EMERGING (10 <= n < 30): Building track record. Show 75% size, no probability.
3. EMPIRICAL (30 <= n < 100): Historical stats. Show win rate, Wilson CI (80%), edge check, no probability.
4. CALIBRATED (n >= 100): Calibrated probability + Brier + Kelly sizing (only if validation Brier < 0.20).
"""

import math
import json
import hashlib
from typing import Optional
from backend.db import get_db, get_setting


def compute_fingerprint(signal_types: list[str], regime: str | None) -> str:
    """Compute a SHA256 hash of sorted signal types + market regime."""
    clean_signals = sorted([str(s).strip() for s in signal_types if s])
    regime_str = (regime or "UNKNOWN").upper().strip()
    raw = "|".join(clean_signals) + ":" + regime_str
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wilson_confidence_interval(wins: int, n: int, z: float = 1.28) -> tuple[float, float]:
    """Wilson score confidence interval at confidence z (1.28 = 80% CI)."""
    if n <= 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


def get_honest_assessment(signals: list[dict], score: float, regime: str | None) -> dict:
    """Assess a recommendation based on historical trade counts of its signal fingerprint.

    Args:
        signals: List of active signal dictionaries containing 'type'.
        score: The recommendation score.
        regime: Current market regime (BULL/BEAR/SIDEWAYS/HIGH_VOL).

    Returns:
        A dictionary containing honest assessment metrics.
    """
    signal_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]
    fingerprint = compute_fingerprint(signal_types, regime)
    
    n_trades = 0
    wins = 0
    win_rate = 0.0
    wilson_lower = 0.0
    wilson_upper = 0.0
    avg_pnl = 0.0
    cached = False

    # 1. Attempt O(1) Cache Lookup
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl 
                   FROM signal_performance_cache WHERE fingerprint = ?""",
                (fingerprint,),
            ).fetchone()
            if row:
                n_trades = row["n_trades"]
                wins = row["wins"]
                win_rate = row["win_rate"]
                wilson_lower = row["wilson_lower"]
                wilson_upper = row["wilson_upper"]
                avg_pnl = row["avg_pnl"]
                cached = True
    except Exception:
        pass

    # 2. Fallback to direct DB query if cache miss
    if not cached:
        try:
            with get_db() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) as n,
                           SUM(CASE WHEN pnl_5d_pct > 0 THEN 1 ELSE 0 END) as wins,
                           AVG(pnl_5d_pct) as avg_pnl
                    FROM (
                        SELECT pnl_5d_pct, signal_fingerprint FROM paper_trades WHERE pnl_5d_pct IS NOT NULL
                        UNION ALL
                        SELECT pnl_5d_pct, signal_fingerprint FROM shadow_trades WHERE pnl_5d_pct IS NOT NULL
                    )
                    WHERE signal_fingerprint = ?
                    """,
                    (fingerprint,),
                ).fetchone()
                if row and row["n"] > 0:
                    n_trades = row["n"]
                    wins = row["wins"] or 0
                    win_rate = wins / n_trades
                    avg_pnl = row["avg_pnl"] or 0.0
                    wilson_lower, wilson_upper = wilson_confidence_interval(wins, n_trades)
        except Exception:
            pass

    # 3. Classify Tier
    abs_score = abs(score)
    
    # Defaults
    tier = "EXPLORATORY"
    message = "Insufficient data — Paper trade only, 50% size"
    suggested_size = 5.0
    probability = None
    brier_score = None
    kelly_pct = None

    if n_trades < 10:
        # Tier 1: EXPLORATORY
        tier = "EXPLORATORY"
        message = "Insufficient data — Paper trade only, 50% size"
        suggested_size = 5.0
    elif n_trades < 30:
        # Tier 2: EMERGING
        tier = "EMERGING"
        message = "Building track record — 75% size, track carefully"
        suggested_size = 7.5
    elif n_trades < 100:
        # Tier 3: EMPIRICAL
        tier = "EMPIRICAL"
        edge_label = "NONE"
        if wilson_lower > 0.50:
            edge_label = "STRONG"
        elif win_rate > 0.50:
            edge_label = "MARGINAL"
        message = f"Historical: {win_rate:.0%} win rate ({wilson_lower:.0%}-{wilson_upper:.0%} CI) — Edge: {edge_label}"
        suggested_size = 10.0
    else:
        # Tier 4: CALIBRATED (check if model is available)
        beta_0 = get_setting("calibration_model_beta_0")
        beta_1 = get_setting("calibration_model_beta_1")
        brier = get_setting("calibration_model_brier")
        
        has_calibrated_model = False
        if beta_0 is not None and beta_1 is not None and brier is not None:
            try:
                b0 = float(beta_0)
                b1 = float(beta_1)
                br = float(brier)
                if br < 0.20:
                    has_calibrated_model = True
                    # Model probability formula: sig(beta_0 + beta_1 * abs_score)
                    logit = b0 + b1 * abs_score
                    p = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
                    probability = round(p * 100, 0)
                    brier_score = round(br, 2)
                    
                    # Kelly sizing: 2p - 1
                    k_frac = 2.0 * p - 1.0
                    kelly_pct = max(0.0, round(k_frac * 100, 1))
                    suggested_size = max(1.0, kelly_pct)  # min 1% size if trade is taken
                    tier = "CALIBRATED"
                    message = f"Model: {probability:.0f}% probability — Brier: {br:.2f} — Kelly: {kelly_pct:.1f}%"
            except Exception:
                pass

        if not has_calibrated_model:
            # Fall back to Tier 3 if calibration is unavailable or failed Brier safety check
            tier = "EMPIRICAL"
            edge_label = "NONE"
            if wilson_lower > 0.50:
                edge_label = "STRONG"
            elif win_rate > 0.50:
                edge_label = "MARGINAL"
            message = f"Historical: {win_rate:.0%} win rate ({wilson_lower:.0%}-{wilson_upper:.0%} CI) — Edge: {edge_label}"
            suggested_size = 10.0

    return {
        "tier": tier,
        "n_trades": n_trades,
        "display_message": message,
        "suggested_position_size_pct": suggested_size,
        "probability": probability,
        "brier_score": brier_score,
        "win_rate": round(win_rate, 3) if n_trades > 0 else None,
        "wilson_ci": (round(wilson_lower, 3), round(wilson_upper, 3)) if n_trades > 0 else None,
        "kelly_pct": kelly_pct,
        "fingerprint": fingerprint,
    }
