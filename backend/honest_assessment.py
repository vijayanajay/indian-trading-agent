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
import re
from datetime import datetime
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


def get_portfolio_drawdown() -> float:
    """Reconstruct the paper trading portfolio equity curve and compute current drawdown from peak."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT id, ticker, entry_datetime, entry_date, status, notes, 
                          pnl_1d_pct, pnl_3d_pct, pnl_5d_pct, pnl_10d_pct, updated_at,
                          position_size_pct, unrealized_pnl_pct
                   FROM paper_trades"""
            ).fetchall()
        if not rows:
            return 0.0
        
        trades = []
        for r in rows:
            entry_str = r["entry_datetime"] or r["entry_date"]
            if not entry_str:
                continue
            # Parse entry datetime
            try:
                entry_dt = datetime.strptime(entry_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    entry_dt = datetime.strptime(entry_str[:10], "%Y-%m-%d")
                except Exception:
                    entry_dt = datetime.min
            
            status = r["status"]
            unrealized_pnl = r["unrealized_pnl_pct"] if r["unrealized_pnl_pct"] is not None else 0.0
            
            # Determine P&L
            pnl = None
            if status == "active":
                pnl = unrealized_pnl
            elif status == "manually_closed" and r["notes"]:
                match = re.search(r"P&L:\s*([\-\d\.]+)%", r["notes"])
                if match:
                    try:
                        pnl = float(match.group(1))
                    except ValueError:
                        pass
            if pnl is None:
                for k in ["pnl_10d_pct", "pnl_5d_pct", "pnl_3d_pct", "pnl_1d_pct"]:
                    if r[k] is not None:
                        pnl = float(r[k])
                        break
            if pnl is None:
                import logging
                logging.warning(
                    f"Trade ID {r['id']} ({r['ticker']}) has NULL P&L and no fallback prices. Skipping from drawdown calculation."
                )
                continue
                
            # Determine exit datetime
            exit_dt = None
            if status in ("expired", "manually_closed") and r["updated_at"]:
                try:
                    exit_dt = datetime.strptime(r["updated_at"], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        exit_dt = datetime.strptime(r["updated_at"][:10], "%Y-%m-%d")
                    except Exception:
                        pass
            if exit_dt is None:
                exit_dt = datetime.now()
                
            # Get position size with fallback
            pos_size = r["position_size_pct"]
            if pos_size is None:
                pos_size = 5.0
                import logging
                logging.warning(
                    f"Trade ID {r['id']} ({r['ticker']}) has NULL position_size_pct. Falling back to conservative 5%."
                )
                
            trades.append({
                "entry": entry_dt,
                "exit": exit_dt,
                "pnl": pnl,
                "position_size_pct": pos_size,
                "unrealized_pnl_pct": unrealized_pnl,
                "status": status
            })
            
        # Sort events chronologically. Entry events before exit events at the same timestamp.
        events = []
        for i, t in enumerate(trades):
            events.append((t["entry"], "entry", i))
            events.append((t["exit"], "exit", i))
            
        events.sort(key=lambda x: (x[0], 0 if x[1] == "entry" else 1))
        
        equity = 100000.0
        cash = equity
        peak_equity = equity
        
        def get_open_value(open_pos):
            val = 0.0
            for o_idx, o_alloc in open_pos.items():
                if trades[o_idx]["status"] == "active":
                    val += o_alloc * (1.0 + trades[o_idx]["unrealized_pnl_pct"] / 100.0)
                else:
                    val += o_alloc
            return val

        open_positions = {}
        for evt_time, evt_type, idx in events:
            if evt_type == "entry":
                current_equity = cash + get_open_value(open_positions)
                alloc = current_equity * (trades[idx]["position_size_pct"] / 100.0)
                cash -= alloc
                open_positions[idx] = alloc
            elif evt_type == "exit":
                if idx in open_positions:
                    alloc = open_positions.pop(idx)
                    pnl_pct = trades[idx]["pnl"]
                    returned = alloc * (1.0 + pnl_pct / 100.0)
                    cash += returned
                    
            current_equity = cash + get_open_value(open_positions)
            if current_equity > peak_equity:
                peak_equity = current_equity
                
        current_equity = cash + get_open_value(open_positions)
        if peak_equity <= 0:
            return 0.0
        drawdown = (peak_equity - current_equity) / peak_equity * 100.0
        return max(0.0, drawdown)
    except Exception:
        return 0.0


def get_honest_assessment(signals: list[dict], score: float, regime: str | None, risk_reward_ratio: float | None = None) -> dict:
    """Assess a recommendation based on historical trade counts of its signal fingerprint.

    Args:
        signals: List of active signal dictionaries containing 'type'.
        score: The recommendation score.
        regime: Current market regime (BULL/BEAR/SIDEWAYS/HIGH_VOL).
        risk_reward_ratio: Optional actual risk reward ratio from the trade plan.

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
            # Prime cache on request if empty
            cache_count = conn.execute("SELECT COUNT(*) as cnt FROM signal_performance_cache").fetchone()
            if cache_count and cache_count["cnt"] == 0:
                try:
                    from backend.cron import recompute_fingerprints_and_features_for_last_180_days
                    recompute_fingerprints_and_features_for_last_180_days()
                except Exception:
                    pass

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
                # Query matches by fingerprint OR NULL fingerprints to check dynamically
                rows = conn.execute(
                    """
                    SELECT source, ticker, entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                    FROM (
                        SELECT 'paper' as source, ticker, entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                        FROM paper_trades
                        WHERE pnl_5d_pct IS NOT NULL
                        UNION ALL
                        SELECT 'shadow' as source, ticker, signal_date as entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                        FROM shadow_trades
                        WHERE pnl_5d_pct IS NOT NULL
                    )
                    WHERE signal_fingerprint = ? OR signal_fingerprint IS NULL
                    """,
                    (fingerprint,),
                ).fetchall()
                
                n_trades = 0
                wins = 0
                sum_pnl = 0.0
                
                unique_trades = {}
                for r in rows:
                    fp = r["signal_fingerprint"]
                    if fp is None:
                        trig = r["triggered_signals"]
                        try:
                            sig_list = json.loads(trig) if trig else []
                        except Exception:
                            sig_list = []
                        if not isinstance(sig_list, list):
                            sig_list = []
                        sig_types = [s.get("type") for s in sig_list if isinstance(s, dict) and s.get("type")]
                        reg = r["regime_at_entry"]
                        fp = compute_fingerprint(sig_types, reg)
                    
                    if fp == fingerprint:
                        key = (r["ticker"], r["entry_date"], fp)
                        if key not in unique_trades or r["source"] == "paper":
                            unique_trades[key] = r

                for r in unique_trades.values():
                    n_trades += 1
                    pnl = r["pnl_5d_pct"]
                    if pnl > 0:
                        wins += 1
                    sum_pnl += pnl

                if n_trades > 0:
                    win_rate = wins / n_trades
                    avg_pnl = sum_pnl / n_trades
                    wilson_lower, wilson_upper = wilson_confidence_interval(wins, n_trades)
        except Exception:
            pass

    # 3. Classify Tier
    abs_score = abs(score)
    
    # Defaults
    tier = "EXPLORATORY"
    message = "Paper trade only — no probability estimate"
    suggested_size = 5.0
    probability = None
    brier_score = None
    kelly_pct = None
    low_confidence = False

    if n_trades < 10:
        # Tier 1: EXPLORATORY
        tier = "EXPLORATORY"
        message = "Paper trade only — no probability estimate"
        suggested_size = 5.0
    elif n_trades < 30:
        # Tier 2: EMERGING
        tier = "EMERGING"
        message = "Building track record — no probability estimate"
        suggested_size = 7.5
    elif n_trades < 100:
        # Tier 3: EMPIRICAL
        tier = "EMPIRICAL"
        win_rate_pct = round(win_rate * 100)
        wilson_lower_pct = round(wilson_lower * 100)
        wilson_upper_pct = round(wilson_upper * 100)
        message = f"Historical win rate: {win_rate_pct}% ({wilson_lower_pct}%-{wilson_upper_pct}% confidence)"
        suggested_size = 10.0
    else:
        # Tier 4: CALIBRATED (check if model is available)
        from backend.signal_model import predict_win_probability, load_model_coefficients
        
        coefs, cv_auc, cv_brier = load_model_coefficients()
        
        has_calibrated_model = False
        if coefs and cv_auc is not None and cv_brier is not None:
            try:
                # AUC > 0.55 and Brier < 0.20 safety check
                if cv_auc > 0.55 and cv_brier < 0.20:
                    p = predict_win_probability(fingerprint, regime, signals=signals)
                    if p is not None:
                        has_calibrated_model = True
                        probability = round(p * 100, 0)
                        br = cv_brier
                        brier_score = round(br, 2)
                    
                    # Query average positive and absolute average negative returns
                    avg_win = 0.0
                    avg_loss = 0.0
                    low_confidence = True
                    try:
                        with get_db() as conn:
                            pnl_rows = conn.execute(
                                """
                                SELECT source, ticker, entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                                FROM (
                                    SELECT 'paper' as source, ticker, entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                                    FROM paper_trades
                                    WHERE pnl_5d_pct IS NOT NULL
                                    UNION ALL
                                    SELECT 'shadow' as source, ticker, signal_date as entry_date, pnl_5d_pct, signal_fingerprint, triggered_signals, regime_at_entry
                                    FROM shadow_trades
                                    WHERE pnl_5d_pct IS NOT NULL
                                )
                                WHERE signal_fingerprint = ? OR signal_fingerprint IS NULL
                                """,
                                (fingerprint,),
                            ).fetchall()
                            
                            wins_list = []
                            losses_list = []
                            unique_pnl_trades = {}
                            for r in pnl_rows:
                                fp = r["signal_fingerprint"]
                                if fp is None:
                                    trig = r["triggered_signals"]
                                    try:
                                        sig_list = json.loads(trig) if trig else []
                                    except Exception:
                                        sig_list = []
                                    if not isinstance(sig_list, list):
                                        sig_list = []
                                    sig_types = [s.get("type") for s in sig_list if isinstance(s, dict) and s.get("type")]
                                    reg = r["regime_at_entry"]
                                    fp = compute_fingerprint(sig_types, reg)
                                
                                if fp == fingerprint:
                                    key = (r["ticker"], r["entry_date"], fp)
                                    if key not in unique_pnl_trades or r["source"] == "paper":
                                        unique_pnl_trades[key] = r
                                        
                            for r in unique_pnl_trades.values():
                                pnl = r["pnl_5d_pct"]
                                if pnl > 0:
                                    wins_list.append(pnl)
                                elif pnl < 0:
                                    losses_list.append(pnl)
                            
                            avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
                            avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0.0
                            if avg_win > 0.0 and avg_loss < 0.0:
                                low_confidence = False
                    except Exception:
                        pass

                    if has_calibrated_model:
                        if low_confidence:
                            kelly_pct = 0.0
                            suggested_size = 0.0
                            message = "DO NOT TRADE (insufficient win/loss data for Kelly sizing)"
                        else:
                            b = risk_reward_ratio if (risk_reward_ratio is not None and risk_reward_ratio > 0) else (avg_win / abs(avg_loss))
                            
                            # Full Kelly formula
                            q = 1.0 - p
                            k_frac = (p * b - q) / b
                            
                            if k_frac < 0.0:
                                kelly_pct = 0.0
                                suggested_size = 0.0
                                message = "DO NOT TRADE"
                            else:
                                kelly_pct = round(k_frac * 100, 1)
                                if kelly_pct > 15.0:
                                    kelly_pct = 15.0
                                
                                suggested_size = max(1.0, kelly_pct)
                                
                                # Check portfolio drawdown ceiling
                                drawdown = get_portfolio_drawdown()
                                if drawdown > 10.0:
                                    kelly_pct = 0.0
                                    suggested_size = 0.0
                                    message = f"DO NOT TRADE (portfolio drawdown > 10%)"
                                else:
                                    message = f"Model: {probability:.0f}% probability — Brier: {br:.2f} — Kelly: {kelly_pct:.1f}%"
                        
                        tier = "CALIBRATED"
            except Exception:
                pass

        if not has_calibrated_model:
            # Fall back to Tier 3 if calibration is unavailable or failed Brier safety check
            tier = "EMPIRICAL"
            probability = None
            win_rate_pct = round(win_rate * 100) if n_trades > 0 else 0
            wilson_lower_pct = round(wilson_lower * 100) if n_trades > 0 else 0
            wilson_upper_pct = round(wilson_upper * 100) if n_trades > 0 else 0
            message = f"Historical win rate: {win_rate_pct}% ({wilson_lower_pct}%-{wilson_upper_pct}% confidence)"
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
        "low_confidence": low_confidence,
    }
