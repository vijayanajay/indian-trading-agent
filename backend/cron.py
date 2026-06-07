"""Background Cron Jobs — database backfill, cache pre-computation, and model training.

Runs on a daemon thread during application lifespan. Exposes functions to be run
periodically or triggered manually via API.
"""

import time
import json
import logging
import threading
from datetime import date, datetime, timedelta
import numpy as np

from backend.db import get_db, get_setting, set_setting
from backend.honest_assessment import compute_fingerprint, wilson_confidence_interval

logger = logging.getLogger("background-cron")


def recompute_fingerprints_and_features_for_last_180_days() -> dict:
    """Backfill fingerprints, regimes, volatility, and FII flow data for closed/active trades.

    Rebuilds the O(1) cache table afterwards.
    """
    logger.info("Starting fingerprint and feature backfill...")
    today = date.today()
    cutoff_date = (today - timedelta(days=180)).isoformat()

    # Import dependencies locally to avoid circular dependencies
    from backend.market_regime import get_cached_regime
    from backend.fii_dii import get_data_for_date

    trades_updated = 0
    shadows_updated = 0

    # 1. Backfill paper_trades
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, ticker, entry_date, triggered_signals, regime_at_entry,
                      fii_flow_at_entry, volatility_at_entry, signal_fingerprint
               FROM paper_trades
               WHERE entry_date >= ?""",
            (cutoff_date,),
        ).fetchall()

    for r in rows:
        updates = {}
        entry_date_str = r["entry_date"]
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        except Exception:
            continue

        # Triggered signals parsing
        signals = []
        if r["triggered_signals"]:
            try:
                signals = json.loads(r["triggered_signals"])
            except Exception:
                pass
        if not isinstance(signals, list):
            signals = []
        
        signal_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]

        # Resolve regime_at_entry
        regime = r["regime_at_entry"]
        if not regime:
            regime_info = get_cached_regime(entry_date)
            regime = regime_info.get("regime")
            if regime and regime != "UNKNOWN":
                updates["regime_at_entry"] = regime

        # Resolve fii_flow_at_entry
        fii_flow = r["fii_flow_at_entry"]
        if not fii_flow:
            fii_info = get_data_for_date(entry_date_str)
            if fii_info and fii_info.get("fii_net") is not None:
                fii_flow = f"{fii_info['fii_net']:.0f} Cr"
                updates["fii_flow_at_entry"] = fii_flow

        # Resolve volatility_at_entry
        volatility = r["volatility_at_entry"]
        if not volatility:
            regime_info = get_cached_regime(entry_date)
            if regime_info.get("annualized_vol_pct") is not None:
                volatility = regime_info["annualized_vol_pct"]
                updates["volatility_at_entry"] = volatility

        # Compute fingerprint
        fingerprint = compute_fingerprint(signal_types, regime)
        if fingerprint != r["signal_fingerprint"]:
            updates["signal_fingerprint"] = fingerprint

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [r["id"]]
            with get_db() as conn:
                conn.execute(f"UPDATE paper_trades SET {set_clause} WHERE id = ?", params)
            trades_updated += 1

    # 2. Backfill shadow_trades
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ticker, signal_date, triggered_signals, regime_at_entry,
                      fii_flow_at_entry, volatility_at_entry, signal_fingerprint
               FROM shadow_trades
               WHERE signal_date >= ?""",
            (cutoff_date,),
        ).fetchall()

    for r in rows:
        updates = {}
        entry_date_str = r["signal_date"]
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        except Exception:
            continue

        signals = []
        if r["triggered_signals"]:
            try:
                signals = json.loads(r["triggered_signals"])
            except Exception:
                pass
        if not isinstance(signals, list):
            signals = []
        
        signal_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]

        regime = r["regime_at_entry"]
        if not regime:
            regime_info = get_cached_regime(entry_date)
            regime = regime_info.get("regime")
            if regime and regime != "UNKNOWN":
                updates["regime_at_entry"] = regime

        fii_flow = r["fii_flow_at_entry"]
        if not fii_flow:
            fii_info = get_data_for_date(entry_date_str)
            if fii_info and fii_info.get("fii_net") is not None:
                fii_flow = f"{fii_info['fii_net']:.0f} Cr"
                updates["fii_flow_at_entry"] = fii_flow

        volatility = r["volatility_at_entry"]
        if not volatility:
            regime_info = get_cached_regime(entry_date)
            if regime_info.get("annualized_vol_pct") is not None:
                volatility = regime_info["annualized_vol_pct"]
                updates["volatility_at_entry"] = volatility

        fingerprint = compute_fingerprint(signal_types, regime)
        if fingerprint != r["signal_fingerprint"]:
            updates["signal_fingerprint"] = fingerprint

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [r["ticker"], entry_date_str]
            with get_db() as conn:
                conn.execute(
                    f"UPDATE shadow_trades SET {set_clause} WHERE ticker = ? AND signal_date = ?",
                    params,
                )
            shadows_updated += 1

    # 3. Rebuild signal_performance_cache
    cache_entries = 0
    try:
        with get_db() as conn:
            # Aggregate all closed paper + shadow trades
            rows = conn.execute(
                """
                SELECT signal_fingerprint,
                       COUNT(*) as n,
                       SUM(CASE WHEN pnl_5d_pct > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(pnl_5d_pct) as avg_pnl
                FROM (
                    SELECT pnl_5d_pct, signal_fingerprint FROM paper_trades WHERE pnl_5d_pct IS NOT NULL
                    UNION ALL
                    SELECT pnl_5d_pct, signal_fingerprint FROM shadow_trades WHERE pnl_5d_pct IS NOT NULL
                )
                WHERE signal_fingerprint IS NOT NULL
                GROUP BY signal_fingerprint
                """
            ).fetchall()

            conn.execute("DELETE FROM signal_performance_cache")
            
            for r in rows:
                fp = r["signal_fingerprint"]
                n = r["n"]
                w = r["wins"] or 0
                wr = w / n
                avg_p = r["avg_pnl"] or 0.0
                low, high = wilson_confidence_interval(w, n)
                
                conn.execute(
                    """INSERT OR REPLACE INTO signal_performance_cache
                       (fingerprint, n_trades, wins, win_rate, wilson_lower, wilson_upper, avg_pnl)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (fp, n, w, wr, low, high, avg_p),
                )
                cache_entries += 1
    except Exception as e:
        logger.error(f"Failed to rebuild performance cache: {e}")

    logger.info(
        f"Backfill finished: {trades_updated} paper updates, {shadows_updated} shadow updates, "
        f"{cache_entries} cache rows written."
    )
    
    set_setting("last_fingerprint_run", datetime.now().isoformat())
    return {
        "trades_updated": trades_updated,
        "shadows_updated": shadows_updated,
        "cache_rows": cache_entries,
    }


def fit_logistic_regression(X_vals: np.ndarray, y_vals: np.ndarray, lr: float = 0.1) -> tuple[float, float]:
    """Train a logistic regression model on absolute score values using Newton-Raphson method."""
    N = len(X_vals)
    if N == 0:
        return 0.0, 0.0

    # Standardize input to avoid overflow/scale issues
    x_mean = np.mean(X_vals)
    x_std = np.std(X_vals) if np.std(X_vals) > 0 else 1.0
    x_norm = (X_vals - x_mean) / x_std

    # Prepend ones column for bias
    X = np.column_stack([np.ones(N), x_norm])
    theta = np.zeros(2)  # [beta_0, beta_1]

    # Iterative Newton-Raphson
    for _ in range(15):
        logits = np.dot(X, theta)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        grad = np.dot(X.T, (p - y_vals)) / N
        W = p * (1.0 - p)
        W = np.clip(W, 1e-5, 1.0)
        
        # Hessian: X.T * diag(W) * X / N
        H = np.dot(X.T * W, X) / N
        try:
            theta -= np.dot(np.linalg.inv(H), grad)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent
            theta -= lr * grad

    # De-standardize parameters back to original scale:
    # beta_0_orig + beta_1_orig * x = beta_0 + beta_1 * (x - mean)/std
    beta_1_orig = theta[1] / x_std
    beta_0_orig = theta[0] - theta[1] * x_mean / x_std
    return float(beta_0_orig), float(beta_1_orig)


def retrain_calibration_model() -> dict:
    """Train logistic regression model on last 12 months, validate on recent 3 months."""
    logger.info("Starting calibration model retraining...")
    today = date.today()
    three_months_ago = (today - timedelta(days=90)).isoformat()
    fifteen_months_ago = (today - timedelta(days=450)).isoformat()

    # 1. Fetch closed trades
    try:
        with get_db() as conn:
            # Shadow trades are ground truth of all recommendations; paper trades are user subset.
            # Query combined dataset where pnl_5d_pct is available
            rows = conn.execute(
                """
                SELECT score, pnl_5d_pct, entry_date FROM (
                    SELECT score, pnl_5d_pct, entry_date FROM paper_trades WHERE pnl_5d_pct IS NOT NULL AND score IS NOT NULL
                    UNION ALL
                    SELECT score, pnl_5d_pct, signal_date as entry_date FROM shadow_trades WHERE pnl_5d_pct IS NOT NULL AND score IS NOT NULL
                )
                WHERE entry_date >= ?
                ORDER BY entry_date ASC
                """,
                (fifteen_months_ago,),
            ).fetchall()
    except Exception as e:
        logger.error(f"Failed to retrieve training data: {e}")
        return {"status": "error", "message": f"Database read failed: {e}"}

    train_X = []
    train_y = []
    val_X = []
    val_y = []

    for r in rows:
        score = abs(r["score"])
        success = 1 if r["pnl_5d_pct"] > 0 else 0
        
        if r["entry_date"] >= three_months_ago:
            val_X.append(score)
            val_y.append(success)
        else:
            train_X.append(score)
            train_y.append(success)

    n_train = len(train_X)
    n_val = len(val_X)
    total_samples = n_train + n_val

    if total_samples < 50:
        logger.warning(f"Insufficient training data: {total_samples} samples. Need at least 50.")
        return {
            "status": "warning",
            "message": f"Insufficient data (n={total_samples}). Need at least 50 closed trades.",
            "n_samples": total_samples,
        }

    # Conver to numpy arrays
    X_train = np.array(train_X)
    y_train = np.array(train_y)
    X_val = np.array(val_X)
    y_val = np.array(val_y)

    # 2. Fit model
    beta_0, beta_1 = fit_logistic_regression(X_train, y_train)

    # 3. Calculate validation Brier score
    if n_val > 0:
        logits_val = beta_0 + beta_1 * X_val
        probs_val = 1.0 / (1.0 + np.exp(-np.clip(logits_val, -15.0, 15.0)))
        brier = float(np.mean((probs_val - y_val) ** 2))
    else:
        # Fall back to training set Brier if validation is empty (safeguard)
        logits_train = beta_0 + beta_1 * X_train
        probs_train = 1.0 / (1.0 + np.exp(-np.clip(logits_train, -15.0, 15.0)))
        brier = float(np.mean((probs_train - y_train) ** 2))

    logger.info(f"Calibration results: beta_0={beta_0:.4f}, beta_1={beta_1:.4f}, Brier={brier:.4f}")

    # 4. Save results to settings
    # Note: Brier safety guard is checked in the HonestAssessmentEngine.
    # We still save the model coefficients, but the engine will bypass it if Brier >= 0.20
    set_setting("calibration_model_beta_0", str(beta_0))
    set_setting("calibration_model_beta_1", str(beta_1))
    set_setting("calibration_model_brier", str(brier))
    set_setting("calibration_model_trained_at", datetime.now().isoformat())
    set_setting("calibration_model_n_trades", str(total_samples))
    set_setting("last_model_retrain_run", datetime.now().isoformat())

    status = "ok" if brier < 0.20 else "failed_brier_threshold"
    msg = "Model updated successfully." if brier < 0.20 else f"Model Brier score too high ({brier:.3f} >= 0.20). Fallback active."
    
    return {
        "status": status,
        "message": msg,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "brier_score": brier,
        "n_train": n_train,
        "n_val": n_val,
    }


def _cron_loop():
    """Background runner loop checking and running tasks."""
    logger.info("Background Cron Daemon started.")
    # Allow system startup to stabilize before running checks
    time.sleep(15)

    while True:
        try:
            now = datetime.now()
            
            # 1. Fingerprint backfill - run daily
            last_fp_run = get_setting("last_fingerprint_run")
            should_run_fp = True
            if last_fp_run:
                try:
                    last_run_dt = datetime.fromisoformat(last_fp_run)
                    if (now - last_run_dt) < timedelta(days=1):
                        should_run_fp = False
                except Exception:
                    pass

            if should_run_fp:
                recompute_fingerprints_and_features_for_last_180_days()

            # 2. Calibration retraining - run weekly
            last_train_run = get_setting("last_model_retrain_run")
            should_run_train = True
            if last_train_run:
                try:
                    last_run_dt = datetime.fromisoformat(last_train_run)
                    if (now - last_run_dt) < timedelta(days=7):
                        should_run_train = False
                except Exception:
                    pass

            if should_run_train:
                retrain_calibration_model()

        except Exception as e:
            logger.error(f"Error in background cron execution: {e}")

        # Sleep for 1 hour
        time.sleep(3600)


def start_cron_daemon():
    """Spawn the daemon thread."""
    t = threading.Thread(target=_cron_loop, daemon=True, name="CronDaemon")
    t.start()
