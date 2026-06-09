import pytest
from unittest.mock import patch
from backend.concentration import get_sector_allocation, check_new_trade_concentration

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
