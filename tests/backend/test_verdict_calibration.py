import pytest
from datetime import date
import pandas as pd
from unittest.mock import patch, MagicMock
from backend.verdict_calibration import _add_trading_days, _classify_outcome, _get_nifty_close_for_date

def test_add_trading_days_basic():
    # Monday 2026-06-08 + 1 trading day = Tuesday 2026-06-09
    start = date(2026, 6, 8)
    assert _add_trading_days(start, 1) == date(2026, 6, 9)

    # Monday 2026-06-08 + 3 trading days = Thursday 2026-06-11
    assert _add_trading_days(start, 3) == date(2026, 6, 11)


def test_add_trading_days_weekend():
    # Friday 2026-06-05 + 1 trading day = Monday 2026-06-08
    start = date(2026, 6, 5)
    assert _add_trading_days(start, 1) == date(2026, 6, 8)

    # Friday 2026-06-05 + 5 trading days = Friday 2026-06-12
    assert _add_trading_days(start, 5) == date(2026, 6, 12)


def test_add_trading_days_nse_holiday():
    # 2026-01-26 (Monday) is Republic Day (NSE holiday).
    # Friday 2026-01-23 + 1 trading day = Tuesday 2026-01-27
    start = date(2026, 1, 23)
    assert _add_trading_days(start, 1) == date(2026, 1, 27)

    # Friday 2026-01-23 + 2 trading days = Wednesday 2026-01-28
    assert _add_trading_days(start, 2) == date(2026, 1, 28)


def test_classify_outcome_green():
    # Green verdict, positive returns above noise floor -> correct
    assert _classify_outcome("GREEN", 0.15) == "predicted_correctly"
    # Green verdict, positive returns below noise floor -> neutral
    assert _classify_outcome("GREEN", 0.05) == "neutral"
    # Green verdict, negative returns -> neutral/wrong?
    # Wait, let's check the code:
    # 198:     if v == "GREEN":
    # 199:         if return_pct > DIRECTIONAL_NOISE_FLOOR:
    # 200:             return "predicted_correctly"
    # 201:         if return_pct < -DIRECTIONAL_NOISE_FLOOR:
    # 202:             return "predicted_wrong"
    # 203:         return "neutral"
    assert _classify_outcome("GREEN", -0.15) == "predicted_wrong"
    assert _classify_outcome("GREEN", -0.05) == "neutral"


def test_classify_outcome_red():
    # Red verdict:
    # 200:     if v == "RED": (Wait, let's verify RED logic in backend/verdict_calibration.py)
    # Let's verify RED logic by testing it. We will check it with the test run.
    # From lines 10-11:
    # - GREEN  predicts UP    → correct if Nifty return > +0.10%, wrong if < -0.10%, else neutral
    # - RED    predicts DOWN  → correct if Nifty return < -0.10%, wrong if > +0.10%, else neutral
    assert _classify_outcome("RED", -0.15) == "predicted_correctly"
    assert _classify_outcome("RED", 0.15) == "predicted_wrong"
    assert _classify_outcome("RED", 0.05) == "neutral"


def test_classify_outcome_yellow():
    # Yellow verdict:
    # 204:     if v == "YELLOW":
    # 205:         # Yellow says "no edge / quiet day" — correct if market stayed inside the band
    # 206:         if abs(return_pct) <= QUIET_DAY_CEILING:
    # 207:             return "predicted_correctly"
    # 208:         return "predicted_wrong"
    assert _classify_outcome("YELLOW", 0.25) == "predicted_correctly"
    assert _classify_outcome("YELLOW", -0.25) == "predicted_correctly"
    assert _classify_outcome("YELLOW", 0.55) == "predicted_wrong"
    assert _classify_outcome("YELLOW", -0.55) == "predicted_wrong"


def test_get_nifty_close_for_date():
    # Create mock history DataFrame
    dates = pd.to_datetime(["2026-06-08", "2026-06-10", "2026-06-12"])
    mock_df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=dates)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df

    with patch("yfinance.Ticker", return_value=mock_ticker):
        # Case 1: Exact match found
        close = _get_nifty_close_for_date("2026-06-10")
        assert close == 110.0

        # Case 2: Exact match not found, direction="backward" (default)
        # Target is 2026-06-09. Prior date is 2026-06-08.
        close_back = _get_nifty_close_for_date("2026-06-09", direction="backward")
        assert close_back == 100.0

        # Case 3: Exact match not found, direction="forward"
        # Target is 2026-06-09. Next date is 2026-06-10.
        close_fwd = _get_nifty_close_for_date("2026-06-09", direction="forward")
        assert close_fwd == 110.0
