import pytest
import pandas as pd
from unittest.mock import patch
from backend.concentration import (
    get_sector_allocation,
    check_new_trade_concentration,
    compute_pairwise_correlation,
    check_correlation_clustering,
)

@patch("backend.concentration.get_analysis_history")
@patch("backend.concentration.list_paper_trades")
def test_get_sector_allocation_with_custom_sizing(mock_list_paper, mock_get_analysis):
    # Mock analysis history to be empty for simplicity
    mock_get_analysis.return_value = []
    
    # Mock active paper trades: one IT stock with 15% size, one Banking stock with no size (fallback)
    mock_list_paper.return_value = [
        {
            "id": 1,
            "ticker": "INFY",
            "direction": "LONG",
            "entry_price": 1500.0,
            "entry_date": "2026-06-08",
            "strategy": "Breakout",
            "position_size_pct": 15.0,
        },
        {
            "id": 2,
            "ticker": "SBIN",
            "direction": "LONG",
            "entry_price": 600.0,
            "entry_date": "2026-06-08",
            "strategy": "Gap Up",
            "position_size_pct": None,  # should fallback to 10%
        }
    ]
    
    total_capital = 500000.0
    result = get_sector_allocation(total_capital=total_capital)
    
    # Assertions on sector summary
    by_sector = result["by_sector"]
    
    # INFY is IT sector (mapped in Cyclical SECTOR_MAP)
    assert "IT" in by_sector
    it_pos = by_sector["IT"]["positions"][0]
    assert it_pos["ticker"] == "INFY"
    # 15% of 500000 = 75000
    assert it_pos["position_value"] == 75000.0
    assert by_sector["IT"]["percent"] == 15.0
    
    # SBIN is Banks sector (mapped in Cyclical SECTOR_MAP)
    assert "Banks" in by_sector
    bank_pos = by_sector["Banks"]["positions"][0]
    assert bank_pos["ticker"] == "SBIN"
    # Fallback to 10% of 500000 = 50000
    assert bank_pos["position_value"] == 50000.0
    assert by_sector["Banks"]["percent"] == 10.0
    
    # Total portfolio assertions
    assert result["total_positions"] == 2
    assert result["total_allocated"] == 75000.0 + 50000.0
    assert result["total_allocated_pct"] == 25.0


@patch("backend.concentration.get_correlation")
@patch("backend.concentration.save_correlation")
@patch("backend.concentration._fetch_returns")
def test_compute_pairwise_correlation(mock_fetch, mock_save, mock_get):
    # Case 1: Cache hit
    mock_get.return_value = 0.55
    res = compute_pairwise_correlation("INFY", "TCS")
    assert res == 0.55
    mock_fetch.assert_not_called()

    # Case 2: Cache miss, fetch returns
    mock_get.return_value = None
    ret_a = pd.Series([float(i) for i in range(25)])
    ret_b = pd.Series([float(i) for i in range(25)])
    
    mock_fetch.side_effect = [ret_a, ret_b]
    
    res = compute_pairwise_correlation("INFY", "TCS")
    # Verify correlation calculation (should be 1.0 since inputs are identical)
    assert res is not None
    assert round(res, 2) == 1.0
    mock_save.assert_called_once()


@patch("backend.concentration.get_open_positions")
@patch("backend.concentration.get_avg_correlation_with_portfolio")
def test_check_correlation_clustering(mock_get_avg, mock_get_pos):
    # Mock open positions
    mock_get_pos.return_value = [
        {"ticker": "TCS"},
        {"ticker": "WIPRO"},
    ]
    
    # Mock average correlation check returns
    mock_get_avg.return_value = {
        "avg_correlation": 0.75,
        "pairwise": {"TCS": 0.78, "WIPRO": 0.72},
        "max_correlation": 0.78,
        "max_correlation_ticker": "TCS",
        "n_computed": 2,
        "n_total": 2,
    }
    
    res = check_correlation_clustering("INFY", threshold=0.70)
    assert res["would_cluster"] is True
    assert res["score_adjustment"] == -1.5  # High correlation cluster (>= 2 partners > 0.70)
    assert len(res["warnings"]) == 1
    assert "High correlation cluster" in res["warnings"][0]

