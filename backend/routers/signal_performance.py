"""Per-signal performance API — track which recommender signals actually win,
and let the user auto-tune the weights from real trade outcomes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.signal_performance import (
    compute_signal_performance,
    apply_tuned_weights,
    reset_tuned_weights,
    get_tuned_weights,
    compute_regime_conditional_weights,
    apply_regime_weights,
    reset_regime_weights,
    get_regime_weights,
)

router = APIRouter(prefix="/api/signal-performance", tags=["signal-performance"])


@router.get("/")
def get_performance(window_days: int = 90):
    """Return per-signal win rate, avg return, and suggested weight delta.

    Args:
        window_days: lookback window for closed trades (default 90).
    """
    return compute_signal_performance(window_days=window_days)


@router.get("/active-weights")
def get_active():
    """Return the currently active tuned-weight overrides (empty if none)."""
    return {"overrides": get_tuned_weights()}


class ApplyRequest(BaseModel):
    window_days: int = 90
    only_keys: Optional[list[str]] = None  # if None, apply all suggested changes


@router.post("/apply")
def apply(req: ApplyRequest):
    """Deprecated: Manual weight overrides are retired."""
    raise HTTPException(
        status_code=400,
        detail="Manual weight overrides are retired. The system uses a trained L1-regularized logistic regression model instead."
    )


@router.post("/reset")
def reset():
    """Deprecated: Manual weight overrides are retired."""
    raise HTTPException(
        status_code=400,
        detail="Manual weight overrides are retired. The system uses a trained L1-regularized logistic regression model instead."
    )


# --- Per-regime conditional weights (Tier 4.1) ---


@router.get("/regime-suggestions")
def regime_suggestions(window_days: int = 180):
    """Compute per-regime weight suggestions (does NOT persist)."""
    return compute_regime_conditional_weights(window_days=window_days)


@router.get("/regime-active")
def regime_active():
    """Return currently-active per-regime overrides from settings."""
    return {"by_regime": get_regime_weights()}


class ApplyRegimeRequest(BaseModel):
    window_days: int = 180
    only_regimes: Optional[list[str]] = None  # e.g., ["HIGH_VOL"] to apply only that one


@router.post("/regime-apply")
def regime_apply(req: ApplyRegimeRequest):
    """Deprecated: Regime weight overrides are retired."""
    raise HTTPException(
        status_code=400,
        detail="Regime weight overrides are retired. The system uses a trained L1-regularized logistic regression model instead."
    )


@router.post("/regime-reset")
def regime_reset():
    """Deprecated: Regime weight overrides are retired."""
    raise HTTPException(
        status_code=400,
        detail="Regime weight overrides are retired. The system uses a trained L1-regularized logistic regression model instead."
    )

