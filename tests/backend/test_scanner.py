import pandas as pd
from backend.scanner import scan_volume_spikes

def test_scan_volume_spikes_normal():
    # 1. Normal case: average volume is positive, current volume is a spike
    stocks_data = [
        {
            "ticker": "TEST1",
            "symbol": "TEST1.NS",
            "hist": pd.DataFrame({"Volume": [100, 100, 100, 100, 300]}),
            "current_close": 105.0,
            "prev_close": 100.0,
            "current_volume": 300,
            "prev_volume": 100
        }
    ]
    results = scan_volume_spikes(stocks_data, multiplier=2.0)
    assert len(results) == 1
    assert results[0]["ticker"] == "TEST1"
    assert results[0]["volume_ratio"] == 3.0
    assert results[0]["direction"] == "BULLISH"

def test_scan_volume_spikes_zero_avg_positive_current():
    # 2. Zero average volume, positive current volume -> should be caught as a spike (vol_ratio = current_volume)
    stocks_data = [
        {
            "ticker": "TEST2",
            "symbol": "TEST2.BO",
            "hist": pd.DataFrame({"Volume": [0, 0, 0, 0, 150]}),
            "current_close": 98.0,
            "prev_close": 100.0,
            "current_volume": 150,
            "prev_volume": 0
        }
    ]
    results = scan_volume_spikes(stocks_data, multiplier=2.0)
    assert len(results) == 1
    assert results[0]["ticker"] == "TEST2"
    assert results[0]["volume_ratio"] == 150.0
    assert results[0]["avg_volume"] == 0
    assert results[0]["direction"] == "BEARISH"

def test_scan_volume_spikes_zero_avg_zero_current():
    # 3. Zero average volume, zero current volume -> should be skipped
    stocks_data = [
        {
            "ticker": "TEST3",
            "symbol": "TEST3.NS",
            "hist": pd.DataFrame({"Volume": [0, 0, 0, 0, 0]}),
            "current_close": 100.0,
            "prev_close": 100.0,
            "current_volume": 0,
            "prev_volume": 0
        }
    ]
    results = scan_volume_spikes(stocks_data, multiplier=2.0)
    assert len(results) == 0
