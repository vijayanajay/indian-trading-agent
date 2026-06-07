"""Confidence calibration API — Brier score + reliability bins for the
recommender's `success_probability` outputs."""

from fastapi import APIRouter

from backend.confidence_calibration import compute_calibration
from backend.cron import retrain_calibration_model, recompute_fingerprints_and_features_for_last_180_days
from backend.db import get_setting

router = APIRouter(prefix="/api/confidence-calibration", tags=["confidence-calibration"])


@router.get("/")
def get_calibration(window_days: int = 180):
    """Returns Brier score + per-bin reliability stats for closed paper_trades."""
    return compute_calibration(window_days=window_days)


@router.get("/model-status")
def get_model_status():
    """Retrieve current calibration model status and coefficients."""
    beta_0 = get_setting("calibration_model_beta_0")
    beta_1 = get_setting("calibration_model_beta_1")
    brier = get_setting("calibration_model_brier")
    trained_at = get_setting("calibration_model_trained_at")
    n_trades = get_setting("calibration_model_n_trades")

    return {
        "has_model": beta_0 is not None and beta_1 is not None and brier is not None,
        "beta_0": float(beta_0) if beta_0 is not None else None,
        "beta_1": float(beta_1) if beta_1 is not None else None,
        "brier_score": float(brier) if brier is not None else None,
        "trained_at": trained_at,
        "n_trades": int(n_trades) if n_trades is not None else None,
        "brier_safety_active": float(brier) >= 0.20 if brier is not None else False,
    }


@router.post("/retrain")
def retrain_model():
    """Manually trigger model retraining."""
    return retrain_calibration_model()


@router.post("/recompute-fingerprints")
def trigger_recompute_fingerprints():
    """Manually trigger fingerprint recomputation and cache rebuild."""
    return recompute_fingerprints_and_features_for_last_180_days()
