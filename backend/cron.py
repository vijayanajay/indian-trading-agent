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
            # Fetch all closed paper + shadow trades with non-null fingerprints
            rows = conn.execute(
                """
                SELECT 'paper' as source, ticker, entry_date, pnl_5d_pct, signal_fingerprint
                FROM paper_trades
                WHERE pnl_5d_pct IS NOT NULL AND signal_fingerprint IS NOT NULL
                UNION ALL
                SELECT 'shadow' as source, ticker, signal_date as entry_date, pnl_5d_pct, signal_fingerprint
                FROM shadow_trades
                WHERE pnl_5d_pct IS NOT NULL AND signal_fingerprint IS NOT NULL
                """
            ).fetchall()

            # Deduplicate by (ticker, entry_date, signal_fingerprint), prioritizing paper trades
            unique_trades = {}
            for r in rows:
                key = (r["ticker"], r["entry_date"], r["signal_fingerprint"])
                if key not in unique_trades or r["source"] == "paper":
                    unique_trades[key] = r

            # Group by signal_fingerprint
            from collections import defaultdict
            fingerprint_groups = defaultdict(list)
            for t in unique_trades.values():
                fp = t["signal_fingerprint"]
                fingerprint_groups[fp].append(t["pnl_5d_pct"])

            conn.execute("DELETE FROM signal_performance_cache")
            
            for fp, pnls in fingerprint_groups.items():
                n = len(pnls)
                w = sum(1 for p in pnls if p > 0)
                wr = w / n if n > 0 else 0.0
                avg_p = sum(pnls) / n if n > 0 else 0.0
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
    """Retrain the L1-regularized logistic regression signal model."""
    logger.info("Starting signal model retraining via cron...")
    from backend.signal_model import train_signal_model
    res = train_signal_model()
    # Mark last train run time in settings
    set_setting("last_model_retrain_run", datetime.now().isoformat())
    return res


def precompute_correlations():
    """Pre-compute correlations for NIFTY 50 pairs daily to warm up the cache."""
    try:
        from backend.concentration import compute_pairwise_correlation, _fetch_returns
        from backend.scanner import NIFTY_100
        import itertools
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Only compute for top 50 to keep it fast (~1225 pairs)
        top_50 = NIFTY_100[:50]
        
        # Fetch returns for all 50 tickers once in parallel to avoid rate limits
        logger.info("Pre-fetching returns for NIFTY 50 tickers in parallel...")
        returns_cache = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_fetch_returns, ticker): ticker for ticker in top_50}
            for f in as_completed(futures):
                ticker = futures[f]
                try:
                    ret = f.result()
                    if ret is not None:
                        returns_cache[ticker] = ret
                except Exception as e:
                    logger.warning(f"Failed to fetch returns for {ticker}: {e}")
        
        # Compute all pairwise combinations using pre-fetched returns sequentially on main thread
        computed = 0
        for a, b in itertools.combinations(top_50, 2):
            ret_a = returns_cache.get(a)
            ret_b = returns_cache.get(b)
            if ret_a is not None and ret_b is not None:
                c = compute_pairwise_correlation(a, b, ret_a, ret_b, fetch_if_missing=False)
                if c is not None:
                    computed += 1
        logger.info(f"Pre-computed {computed} correlations for NIFTY 50.")
    except Exception as e:
        logger.error(f"Correlation pre-computation failed: {e}")



def _cron_loop():
    """Background runner loop checking and running tasks."""
    logger.info("Background Cron Daemon started.")
    # Allow system startup to stabilize before running checks
    time.sleep(15)

    while True:
        try:
            now = datetime.now()
            
            # Check and close trades that hit stop loss
            try:
                from backend.simulation import check_and_trigger_stop_losses
                logger.info("Checking paper trades for stop-loss breaches...")
                triggered = check_and_trigger_stop_losses()
                if triggered > 0:
                    logger.info(f"Triggered stop-loss exits for {triggered} trade(s).")
            except Exception as e:
                logger.error(f"Error checking stop-losses in cron: {e}")

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

            # 1b. Correlation cache pre-computation - run daily
            last_corr_run = get_setting("last_correlation_precompute_run")
            should_run_corr = True
            if last_corr_run:
                try:
                    last_run_dt = datetime.fromisoformat(last_corr_run)
                    if (now - last_run_dt) < timedelta(days=1):
                        should_run_corr = False
                except Exception:
                    pass

            if should_run_corr:
                precompute_correlations()
                set_setting("last_correlation_precompute_run", now.isoformat())

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
            else:
                try:
                    from backend.signal_model import check_and_trigger_retraining
                    check_and_trigger_retraining()
                except Exception as e:
                    logger.error(f"Error checking retraining trigger in cron: {e}")

        except Exception as e:
            logger.error(f"Error in background cron execution: {e}")

        # Sleep for 1 hour
        time.sleep(3600)


def start_cron_daemon():
    """Spawn the daemon thread."""
    t = threading.Thread(target=_cron_loop, daemon=True, name="CronDaemon")
    t.start()
