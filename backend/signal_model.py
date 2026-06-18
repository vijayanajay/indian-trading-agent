"""Signal Model module — offline training, online inference, and feature vectorization."""

import math
import json
import logging
import time
import threading
from datetime import datetime
import numpy as np

from backend.db import get_db, get_setting, set_setting

logger = logging.getLogger("signal-model")

# List of 17 signals from signal_performance.SIGNAL_TYPE_TO_KEY keys/values
SIGNAL_KEYS = [
    "gap_up_filled",
    "gap_up_open",
    "gap_down_filled",
    "gap_down_open",
    "volume_bullish",
    "volume_bearish",
    "breakout_vol_confirmed",
    "breakout_weak",
    "breakdown_support",
    "near_support",
    "near_resistance",
    "rsi_oversold",
    "rsi_overbought",
    "cyclical_bullish",
    "cyclical_bearish",
    "uptrend_strong",
    "downtrend_strong",
]

# Mapping from human-readable signal 'type' to SIGNAL_KEYS
# Must match backend/signal_performance.py SIGNAL_TYPE_TO_KEY
SIGNAL_TYPE_TO_KEY = {
    "Gap Up (Filled)": "gap_up_filled",
    "Gap Up (Unfilled)": "gap_up_open",
    "Gap Down (Filled)": "gap_down_filled",
    "Gap Down (Unfilled)": "gap_down_open",
    "Volume Spike (Bullish)": "volume_bullish",
    "Volume Spike (Bearish)": "volume_bearish",
    "Breakout (Volume Confirmed)": "breakout_vol_confirmed",
    "Breakout (Weak Volume)": "breakout_weak",
    "Breakdown Below Support": "breakdown_support",
    "Near Major Support": "near_support",
    "Near Major Resistance": "near_resistance",
    "RSI Oversold": "rsi_oversold",
    "RSI Overbought": "rsi_overbought",
    "Cyclical (Bullish Month)": "cyclical_bullish",
    "Cyclical (Bearish Month)": "cyclical_bearish",
    "Strong Uptrend": "uptrend_strong",
    "Strong Downtrend": "downtrend_strong",
}

REGIMES = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]

# Curated interactions to prevent overfitting
INTERACTIONS = [
    ("breakout_vol_confirmed", "volume_bullish"),
    ("breakout_weak", "volume_bullish"),
    ("gap_up_filled", "uptrend_strong"),
    ("gap_down_filled", "downtrend_strong"),
    ("near_support", "rsi_oversold"),
    ("near_resistance", "rsi_overbought"),
]

# Order of features in matrices
FEATURE_NAMES = ["intercept"]
FEATURE_NAMES += [f"sig_{k}" for k in SIGNAL_KEYS]
FEATURE_NAMES += [f"regime_{r}" for r in REGIMES]
FEATURE_NAMES += [f"int_{k1}_x_{k2}" for k1, k2 in INTERACTIONS]

# In-memory coefficients cache
_MODEL_CACHE = None  # Tuple of (coefs_dict, auc, brier, last_loaded_time)
_CACHE_LOCK = threading.Lock()
_RETRAIN_LOCK = threading.Lock()


def extract_features(signals: list, regime: str | None) -> dict[str, float]:
    """Turn a list of signals (dicts or strings) and regime into a binary feature dictionary."""
    active_keys = set()
    for s in signals:
        if isinstance(s, dict):
            t = s.get("type")
        else:
            t = s
        if t in SIGNAL_TYPE_TO_KEY:
            active_keys.add(SIGNAL_TYPE_TO_KEY[t])
        elif t in SIGNAL_KEYS:
            active_keys.add(t)

    feats = {"intercept": 1.0}
    for k in SIGNAL_KEYS:
        feats[f"sig_{k}"] = 1.0 if k in active_keys else 0.0

    regime_upper = (regime or "UNKNOWN").upper().strip()
    for r in REGIMES:
        feats[f"regime_{r}"] = 1.0 if r == regime_upper else 0.0

    for k1, k2 in INTERACTIONS:
        feats[f"int_{k1}_x_{k2}"] = 1.0 if (k1 in active_keys and k2 in active_keys) else 0.0

    return feats


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def soft_threshold(x, lmbda):
    return np.sign(x) * np.maximum(0.0, np.abs(x) - lmbda)


def fit_l1_logistic_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0.05, lr: float = 0.1, max_iter: int = 2000, tol: float = 1e-5) -> np.ndarray:
    """Proximal Gradient Descent solver for L1-regularized logistic regression."""
    N, D = X.shape
    w = np.zeros(D)

    for _ in range(max_iter):
        w_old = w.copy()
        p = sigmoid(np.dot(X, w))
        grad = np.dot(X.T, (p - y)) / N

        w_new = w - lr * grad

        w[0] = w_new[0]  # Intercept not regularized
        w[1:] = soft_threshold(w_new[1:], lr * alpha)

        if np.linalg.norm(w - w_old) < tol:
            break

    return w


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate the Area Under the ROC Curve (AUC)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return 0.5

    correct_pairs = 0
    total_pairs = len(pos_indices) * len(neg_indices)
    for p_idx in pos_indices:
        p_val = y_prob[p_idx]
        correct_pairs += np.sum(p_val > y_prob[neg_indices])
        correct_pairs += 0.5 * np.sum(p_val == y_prob[neg_indices])

    return float(correct_pairs / total_pairs)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score (MSE)."""
    return float(np.mean((y_prob - y_true) ** 2))


def get_signals_by_fingerprint(fingerprint: str) -> list:
    """Look up triggered signals for a given fingerprint from database history."""
    try:
        with get_db() as conn:
            # Query paper_trades
            row = conn.execute(
                "SELECT triggered_signals FROM paper_trades WHERE signal_fingerprint = ? AND triggered_signals IS NOT NULL LIMIT 1",
                (fingerprint,),
            ).fetchone()
            if not row:
                # Query shadow_trades
                row = conn.execute(
                    "SELECT triggered_signals FROM shadow_trades WHERE signal_fingerprint = ? AND triggered_signals IS NOT NULL LIMIT 1",
                    (fingerprint,),
                ).fetchone()

            if row and row["triggered_signals"]:
                return json.loads(row["triggered_signals"])
    except Exception as e:
        logger.error(f"Failed to lookup signals by fingerprint {fingerprint}: {e}")
    return []


def load_model_coefficients() -> tuple[dict[str, float], float | None, float | None]:
    """Retrieve model weights from SQLite database."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT feature, coefficient, auc, brier FROM model_coefficients"
            ).fetchall()
        if not rows:
            return {}, None, None
        coefs = {r["feature"]: r["coefficient"] for r in rows}
        auc = rows[0]["auc"]
        brier = rows[0]["brier"]
        return coefs, auc, brier
    except Exception as e:
        logger.error(f"Failed to load model coefficients: {e}")
        return {}, None, None


def get_cached_coefficients() -> tuple[dict[str, float], float | None, float | None]:
    """Get coefficients with a simple 10-second in-memory cache."""
    global _MODEL_CACHE
    now = time.time()
    with _CACHE_LOCK:
        if _MODEL_CACHE is not None and (now - _MODEL_CACHE[3]) < 10.0:
            return _MODEL_CACHE[0], _MODEL_CACHE[1], _MODEL_CACHE[2]

        coefs, auc, brier = load_model_coefficients()
        _MODEL_CACHE = (coefs, auc, brier, now)
        return coefs, auc, brier


def predict_win_probability(fingerprint: str, regime: str | None, signals: list = None) -> float | None:
    """Predict win probability using active coefficients.
    
    If signals is not provided, looks them up using fingerprint hash from DB.
    Returns None if model is cold/failed validation.
    """
    coefs, auc, brier = get_cached_coefficients()
    if not coefs or auc is None or brier is None:
        return None
    if auc <= 0.55 or brier >= 0.20:
        return None

    if signals is None:
        signals = get_signals_by_fingerprint(fingerprint)
        if not signals:
            return None

    feats = extract_features(signals, regime)
    logit = 0.0
    for name, val in feats.items():
        logit += val * coefs.get(name, 0.0)

    prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
    return prob


def count_closed_trades() -> int:
    """Count unique closed paper/shadow trades."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM (
                    SELECT ticker, entry_date FROM paper_trades WHERE pnl_5d_pct IS NOT NULL OR realized_pnl_pct IS NOT NULL
                    UNION
                    SELECT ticker, signal_date as entry_date FROM shadow_trades WHERE pnl_5d_pct IS NOT NULL
                )
                """
            ).fetchone()
            return rows["cnt"] if rows else 0
    except Exception as e:
        logger.error(f"Failed to count closed trades: {e}")
        return 0


def check_and_trigger_retraining():
    """Trigger background retraining if 20+ new closed trades accumulated."""
    if _RETRAIN_LOCK.locked():
        return

    current_count = count_closed_trades()
    if current_count < 50:
        return

    last_trained_str = get_setting("model_last_trained_trade_count")
    last_trained_count = int(last_trained_str) if last_trained_str else 0

    if current_count - last_trained_count >= 20:
        logger.info(f"Triggering background model retraining: {current_count} closed trades (last trained at {last_trained_count})")
        t = threading.Thread(target=train_signal_model, daemon=True, name="SignalModelRetrainer")
        t.start()


def train_signal_model() -> dict:
    """Train the model, run 5-fold CV, and promote to active if validation criteria are met."""
    if not _RETRAIN_LOCK.acquire(blocking=False):
        logger.warning("Signal model retraining already in progress. Skipping.")
        return {"status": "warning", "message": "Retraining already in progress."}
    try:
        return _train_signal_model_internal()
    finally:
        _RETRAIN_LOCK.release()


def _train_signal_model_internal() -> dict:
    """Internal implementation of signal model retraining."""
    logger.info("Starting signal model retraining job...")
    try:
        # 1. Fetch all closed trades with their inputs & outcomes
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT 'paper' as source, ticker, entry_date, triggered_signals, regime_at_entry, 
                       (CASE WHEN status != 'active' THEN COALESCE(realized_pnl_pct, pnl_5d_pct) ELSE pnl_5d_pct END) as pnl_5d_pct,
                       signal_fingerprint
                FROM paper_trades
                WHERE (realized_pnl_pct IS NOT NULL OR pnl_5d_pct IS NOT NULL) AND triggered_signals IS NOT NULL
                UNION ALL
                SELECT 'shadow' as source, ticker, signal_date as entry_date, triggered_signals, regime_at_entry, pnl_5d_pct, signal_fingerprint
                FROM shadow_trades
                WHERE pnl_5d_pct IS NOT NULL AND triggered_signals IS NOT NULL
                """
            ).fetchall()
    except Exception as e:
        logger.error(f"Failed to query training data: {e}")
        return {"status": "error", "message": f"Database query failed: {e}"}

    # De-duplicate in Python deterministically: prefer paper over shadow
    unique_trades = {}
    for r in rows:
        fp = r["signal_fingerprint"]
        if not fp:
            try:
                signals = json.loads(r["triggered_signals"]) if isinstance(r["triggered_signals"], str) else r["triggered_signals"]
            except Exception:
                signals = []
            if not isinstance(signals, list):
                signals = []
            sig_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]
            from backend.honest_assessment import compute_fingerprint
            fp = compute_fingerprint(sig_types, r["regime_at_entry"])

        key = (r["ticker"], r["entry_date"], fp)
        if key not in unique_trades or r["source"] == "paper":
            unique_trades[key] = r

    trades = list(unique_trades.values())
    n_samples = len(trades)
    if n_samples < 50:
        logger.warning(f"Insufficient training data: {n_samples} samples. Need at least 50.")
        return {
            "status": "warning",
            "message": f"Insufficient data (n={n_samples}). Need at least 50 closed trades.",
            "n_samples": n_samples,
        }

    # Prepare features and target
    X_list = []
    y_list = []

    for t in trades:
        try:
            signals = json.loads(t["triggered_signals"]) if isinstance(t["triggered_signals"], str) else t["triggered_signals"]
        except Exception:
            continue
        if not isinstance(signals, list):
            continue

        feats = extract_features(signals, t["regime_at_entry"])
        vec = [feats.get(name, 0.0) for name in FEATURE_NAMES]
        X_list.append(vec)
        
        success = 1.0 if t["pnl_5d_pct"] > 0 else 0.0
        y_list.append(success)

    X = np.array(X_list)
    y = np.array(y_list)

    if len(y) < 50:
        logger.warning(f"Insufficient valid parsed training data: {len(y)} samples.")
        return {
            "status": "warning",
            "message": f"Insufficient parsed data (n={len(y)}). Need at least 50.",
            "n_samples": len(y),
        }

    # 2. 5-Fold Cross-Validation
    k = 5
    indices = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    oof_probs = np.zeros(len(y))
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val = X[val_indices]

        # Fit model on training fold
        w_fold = fit_l1_logistic_regression(X_train, y_train, alpha=0.05, lr=0.1)
        oof_probs[val_indices] = sigmoid(np.dot(X_val, w_fold))

    cv_auc = compute_auc(y, oof_probs)
    cv_brier = compute_brier(y, oof_probs)

    logger.info(f"Cross-Validation results: AUC={cv_auc:.4f}, Brier={cv_brier:.4f}")

    # 3. Validation Check
    if cv_auc <= 0.55 or cv_brier >= 0.20:
        msg = f"Model failed validation checks: CV AUC {cv_auc:.3f} <= 0.55 or Brier {cv_brier:.3f} >= 0.20. Keeping previous model."
        logger.warning(msg)
        # Update settings count to prevent constant retraining triggers
        set_setting("model_last_trained_trade_count", str(n_samples))
        return {
            "status": "validation_failed",
            "message": msg,
            "auc": cv_auc,
            "brier": cv_brier,
            "n_samples": n_samples,
        }

    # 4. Fit final model on all data
    w_final = fit_l1_logistic_regression(X, y, alpha=0.05, lr=0.1)

    # 5. Persist coefficients
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM model_coefficients")
            for name, coef in zip(FEATURE_NAMES, w_final):
                conn.execute(
                    """INSERT INTO model_coefficients (feature, coefficient, auc, brier, last_trained_date)
                       VALUES (?, ?, ?, ?, ?)""",
                    (name, float(coef), cv_auc, cv_brier, datetime.now().isoformat()),
                )
        set_setting("model_last_trained_trade_count", str(n_samples))
        
        # Clear cache to force reload
        global _MODEL_CACHE
        with _CACHE_LOCK:
            _MODEL_CACHE = None

        logger.info(f"Model updated successfully and promoted to active. Coefficients: {dict(zip(FEATURE_NAMES, w_final))}")
        return {
            "status": "success",
            "message": "Model trained and promoted to active.",
            "auc": cv_auc,
            "brier": cv_brier,
            "n_samples": n_samples,
        }
    except Exception as e:
        logger.error(f"Failed to persist model coefficients: {e}")
        return {"status": "error", "message": f"Failed to persist model coefficients: {e}"}
