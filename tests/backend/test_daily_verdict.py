import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from backend.daily_verdict import compute_daily_verdict

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_today_events")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_green_aggressive(mock_recommend, mock_conc, mock_events, mock_today_events, mock_bias):
    mock_bias.return_value = {"bias": "BULLISH", "confidence": "HIGH", "today_fii_net": 5000}
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    # 3 HIGH conviction setups -> favorable flag
    mock_recommend.return_value = {
        "strong_buys": [{"confidence": "HIGH"}, {"confidence": "HIGH"}, {"confidence": "HIGH"}],
        "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "GREEN"
    assert res["label"] == "TRADE"
    assert res["recommended_position_size_pct"] == 1.0
    assert len(res["favorable_flags"]) == 2 # FII buy, setups
    assert len(res["caution_flags"]) == 0

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_red_stand_down(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BEARISH", "confidence": "HIGH", "today_fii_net": -5000}

    # FOMC tomorrow, Expiry today -> 2 flags
    today = date.today()
    mock_events.return_value = [
        {"type": "FOMC", "date": today.isoformat()},
        {"type": "FNO_EXPIRY", "date": today.isoformat()}
    ]

    mock_conc.return_value = {"risk_level": "HIGH", "concentrated_sectors": ["IT"]}

    # 0 setups -> 1 flag
    mock_recommend.return_value = {
        "strong_buys": [], "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "RED"
    assert res["label"] == "STAND DOWN"
    assert res["recommended_position_size_pct"] == 0.0
    assert len(res["caution_flags"]) >= 3 # Bias, FOMC, Expiry, Conc, 0 Setups

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_yellow_selective(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BEARISH", "confidence": "LOW"} # 1 caution flag
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}

    # 1 setup -> no flag, so favorable=0, caution=1
    mock_recommend.return_value = {
        "strong_buys": [{"confidence": "HIGH"}], "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "YELLOW"
    assert res["label"] == "SELECTIVE"
    assert res["recommended_position_size_pct"] == 0.5
    assert len(res["caution_flags"]) == 1

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_yellow_selective_mixed(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "MIXED"} # 1 caution
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}

    # 3 setups -> 1 favorable
    mock_recommend.return_value = {
        "strong_buys": [{"confidence": "HIGH"}, {"confidence": "HIGH"}, {"confidence": "HIGH"}],
        "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "YELLOW"
    assert res["recommended_position_size_pct"] == 0.75
    assert len(res["caution_flags"]) == 1
    assert len(res["favorable_flags"]) == 1

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_green_normal(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BULLISH", "confidence": "LOW"} # 1 favorable
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    mock_recommend.return_value = {
        "strong_buys": [{"confidence": "HIGH"}], # 0 favorable, 0 caution
        "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "GREEN"
    assert res["label"] == "TRADE"
    assert res["recommended_position_size_pct"] == 1.0
    assert len(res["caution_flags"]) == 0
    assert len(res["favorable_flags"]) == 1

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_yellow_quiet(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "NEUTRAL", "confidence": "LOW"} # 0
    mock_events.return_value = [] # 0
    mock_conc.return_value = {"risk_level": "LOW"} # 0
    mock_recommend.return_value = {
        "strong_buys": [{"confidence": "HIGH"}], # 0
        "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "YELLOW"
    assert res["recommended_position_size_pct"] == 0.75
    assert len(res["caution_flags"]) == 0
    assert len(res["favorable_flags"]) == 0

@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_exception_fallback(mock_recommend, mock_conc, mock_events, mock_bias):
    # If all services fail, we should get 0 flags and default to RED/STAND DOWN due to recommender failure
    mock_bias.side_effect = Exception("DB Error")
    mock_events.side_effect = Exception("API Error")
    mock_conc.side_effect = Exception("DB Error")
    mock_recommend.side_effect = Exception("Agent Error")

    res = compute_daily_verdict()
    assert res["verdict"] == "RED"
    assert res["label"] == "STAND DOWN"
    assert res["min_conviction_required"] == "HIGH"
    assert res["filter_results"]["fii_dii"] is None

def test_events_parsing():
    # specifically test the various days_until logic in compute_daily_verdict
    from backend.daily_verdict import compute_daily_verdict
    with patch("backend.fii_dii.get_market_bias") as mock_bias, \
         patch("backend.calendar_data.get_market_events_in_range") as mock_events, \
         patch("backend.concentration.get_concentration_summary") as mock_conc, \
         patch("backend.recommender.recommend") as mock_recommend:

        mock_bias.return_value = {"bias": "NEUTRAL"}
        mock_conc.return_value = {"risk_level": "LOW"}
        mock_recommend.return_value = {"strong_buys": [], "buys": [], "sells": []}

        today = date.today()
        tomorrow = today + timedelta(days=1)
        two_days = today + timedelta(days=2)

        mock_events.return_value = [
            {"type": "BUDGET", "date": tomorrow.isoformat()},
            {"type": "RBI_POLICY", "date": tomorrow.isoformat()},
            {"type": "FOMC", "date": two_days.isoformat()},
            {"type": "FNO_EXPIRY", "date": today.isoformat()}
        ]

        res = compute_daily_verdict()
        assert any("Union Budget" in c for c in res["caution_flags"])
        assert any("RBI Policy tomorrow" in c for c in res["caution_flags"])
        assert any("Fed FOMC in 2 days" in c for c in res["caution_flags"])
        assert any("F&O monthly expiry TODAY" in c for c in res["caution_flags"])

        # Test 0 days
        mock_events.return_value = [
            {"type": "RBI_POLICY", "date": today.isoformat()},
            {"type": "FOMC", "date": today.isoformat()},
            {"type": "BUDGET", "date": today.isoformat()}
        ]

        res = compute_daily_verdict()
        assert any("Union Budget in 0 days" in c for c in res["caution_flags"])
        assert any("RBI Policy decision TODAY" in c for c in res["caution_flags"])
        assert any("Fed FOMC decision TODAY" in c for c in res["caution_flags"])


@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_recommender_failed_forces_zero_trades(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BULLISH", "confidence": "LOW"}
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    mock_recommend.side_effect = Exception("yfinance API down")

    res = compute_daily_verdict()
    assert res["verdict"] == "RED"
    assert res["label"] == "STAND DOWN"
    assert res["max_trades_today"] == 0
    assert res["recommended_position_size_pct"] == 0.0
    assert res["min_conviction_required"] == "HIGH"
    assert len(res["caution_flags"]) == 0
    assert "Recommender unavailable" in res["action"]
    assert "Recommender unavailable" in res["reasoning"]


@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_no_setups_downgrades_green_verdict(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BULLISH", "confidence": "LOW"}
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    mock_recommend.return_value = {
        "strong_buys": [], "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "YELLOW"
    assert res["max_trades_today"] == 3
    assert res["recommended_position_size_pct"] == 0.75


@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_recommender_filters_disabled(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "NEUTRAL", "confidence": "LOW"}
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    mock_recommend.return_value = {
        "strong_buys": [], "buys": [], "sells": []
    }

    compute_daily_verdict()
    
    mock_recommend.assert_called_once_with(
        universe="nifty50",
        min_signals=2,
        apply_market_bias=False,
        apply_event_filter=False,
        apply_concentration_check=False,
        apply_correlation_check=False,
        fetch_if_missing=False
    )


@patch("backend.fii_dii.get_market_bias")
@patch("backend.calendar_data.get_market_events_in_range")
@patch("backend.concentration.get_concentration_summary")
@patch("backend.recommender.recommend")
def test_compute_daily_verdict_all_tickers_failed_triggers_red_verdict(mock_recommend, mock_conc, mock_events, mock_bias):
    mock_bias.return_value = {"bias": "BULLISH", "confidence": "LOW"}
    mock_events.return_value = []
    mock_conc.return_value = {"risk_level": "LOW"}
    mock_recommend.return_value = {
        "universe": "nifty50",
        "total_analyzed": 50,
        "total_with_signals": 0,
        "failed_tickers": [f"TICKER{i}" for i in range(50)],
        "strong_buys": [], "buys": [], "sells": []
    }

    res = compute_daily_verdict()
    assert res["verdict"] == "RED"
    assert res["label"] == "STAND DOWN"
    assert res["max_trades_today"] == 0
    assert res["recommended_position_size_pct"] == 0.0
    assert "Recommender unavailable" in res["action"]


