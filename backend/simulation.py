"""Paper Trading Simulation + Historical Recommendation Backtest.

All simulations are FREE (pure price math from yfinance, no AI API calls).
"""

import yfinance as yf
import uuid
import numpy as np
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from tradingagents.utils.ticker import normalize_ticker
from tradingagents.utils.market_calendar import next_trading_day, is_trading_day, count_trading_days
from backend.scanner import UNIVERSES
from backend.db import (
    add_paper_trade,
    list_paper_trades,
    update_paper_trade_prices,
    update_paper_trade_status,
    save_recommender_backtest_row,
    get_db,
)


# ============================================================
# PAPER TRADING — track virtual positions from recommendations
# ============================================================

SOURCE_STRATEGY_MAP = {
    "recommendation": "Recommendation Engine (combined signals)",
    "scanner": "Market Scanner",
    "ai_analysis": "AI Multi-Agent Pipeline",
    "manual": "Manual Entry",
    "test": "Test",
}


def open_paper_trade(
    ticker: str,
    source: str = "manual",
    strategy: str | None = None,
    signal: str = None,
    score: float = None,
    confidence: str | None = None,
    success_probability: int = None,
    triggered_signals: list | None = None,
    notes: str = None,
    position_size_pct: float | None = None,
    stop_loss_price: float | None = None,
    risk_reward_ratio: float | None = None,
) -> dict:
    """Open a new paper trade at current market price."""
    symbol = normalize_ticker(ticker)
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2d")
        if hist.empty:
            return {"ok": False, "error": f"No price data for {symbol}"}
        current_price = float(hist.iloc[-1]["Close"])
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Determine direction from signal
    direction = "SHORT" if signal and signal.upper() in ("SELL", "STRONG SELL", "UNDERWEIGHT", "SHORT") else "LONG"

    # Auto-populate stop_loss_price and risk_reward_ratio if not provided
    if stop_loss_price is None or risk_reward_ratio is None:
        try:
            from backend.recommender import _analyze_stock
            analysis = _analyze_stock(ticker)
            if analysis:
                if stop_loss_price is None:
                    stop_loss_price = analysis.get("suggested_stop_loss")
                if risk_reward_ratio is None:
                    risk_reward_ratio = analysis.get("risk_reward_ratio")
        except Exception as e:
            print(f"[Simulation] Failed to auto-populate stop loss / RR: {e}", flush=True)

    # Auto-populate strategy name from source if not given
    if not strategy:
        strategy = SOURCE_STRATEGY_MAP.get(source, source)

    trade_id = add_paper_trade({
        "ticker": ticker.upper(),
        "source": source,
        "strategy": strategy,
        "direction": direction,
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "success_probability": success_probability,
        "triggered_signals": triggered_signals,
        "entry_price": round(current_price, 2),
        "notes": notes,
        "position_size_pct": position_size_pct,
        "stop_loss_price": stop_loss_price,
        "risk_reward_ratio": risk_reward_ratio,
    })

    return {
        "ok": True,
        "trade_id": trade_id,
        "ticker": ticker.upper(),
        "direction": direction,
        "entry_price": round(current_price, 2),
        "strategy": strategy,
        "stop_loss_price": stop_loss_price,
        "risk_reward_ratio": risk_reward_ratio,
    }


def close_paper_trade(trade_id: int) -> dict:
    """Close a paper trade at current market price and compute final P&L."""
    trades = list_paper_trades()
    trade = next((t for t in trades if t["id"] == trade_id), None)
    if not trade:
        return {"ok": False, "error": "Trade not found"}

    symbol = normalize_ticker(trade["ticker"])
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2d")
        if hist.empty:
            return {"ok": False, "error": f"No price data for {symbol}"}
        current_price = round(float(hist.iloc[-1]["Close"]), 2)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    entry = trade["entry_price"]
    direction = trade.get("direction", "LONG")
    multiplier = 1 if direction == "LONG" else -1
    pnl_pct = round(multiplier * (current_price - entry) / entry * 100, 2) if entry else 0

    # Update the trade — store close price in the latest available horizon column
    with get_db() as conn:
        conn.execute(
            """UPDATE paper_trades SET
                status = 'manually_closed',
                unrealized_pnl_pct = 0.0,
                notes = COALESCE(notes, '') || '\nClosed at Rs.' || ? || ' on ' || date('now') || '. P&L: ' || ? || '%',
                updated_at = datetime('now')
               WHERE id = ?""",
            (current_price, pnl_pct, trade_id),
        )

    # Also refresh any pending horizon prices
    refresh_paper_trade_prices(trade_id)
    update_paper_trade_status(trade_id, "manually_closed")

    return {
        "ok": True,
        "trade_id": trade_id,
        "ticker": trade["ticker"],
        "entry_price": entry,
        "close_price": current_price,
        "pnl_pct": pnl_pct,
        "direction": direction,
    }


def hit_paper_trade_stop(trade_id: int, current_price: float = None) -> dict:
    """Close a paper trade because its stop-loss was hit."""
    trades = list_paper_trades()
    trade = next((t for t in trades if t["id"] == trade_id), None)
    if not trade:
        return {"ok": False, "error": "Trade not found"}

    entry = trade["entry_price"]
    sl = trade.get("stop_loss_price")
    if not sl:
        return {"ok": False, "error": "No stop-loss defined for this trade"}

    direction = trade.get("direction", "LONG")
    
    # Close at the stop loss price (or current price if provided)
    close_price = round(current_price if current_price is not None else sl, 2)
    
    multiplier = 1 if direction == "LONG" else -1
    pnl_pct = round(multiplier * (close_price - entry) / entry * 100, 2) if entry else 0

    with get_db() as conn:
        conn.execute(
            """UPDATE paper_trades SET
                status = 'hit_stop',
                unrealized_pnl_pct = 0.0,
                notes = COALESCE(notes, '') || '\nStop-loss hit at Rs.' || ? || ' on ' || date('now') || '. P&L: ' || ? || '%',
                updated_at = datetime('now')
               WHERE id = ?""",
            (close_price, pnl_pct, trade_id),
        )

    # Refresh prices and set status
    refresh_paper_trade_prices(trade_id)
    update_paper_trade_status(trade_id, "hit_stop")

    return {
        "ok": True,
        "trade_id": trade_id,
        "ticker": trade["ticker"],
        "entry_price": entry,
        "close_price": close_price,
        "pnl_pct": pnl_pct,
        "direction": direction,
    }


def check_and_trigger_stop_losses() -> int:
    """Check all active paper trades and trigger hit_stop if price breached stop-loss."""
    trades = list_paper_trades(status="active")
    triggered_count = 0
    for trade in trades:
        sl = trade.get("stop_loss_price")
        if not sl:
            continue
        ticker = trade["ticker"]
        symbol = normalize_ticker(ticker)
        direction = trade.get("direction", "LONG")
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="1d")
            if hist.empty:
                continue
            
            current_low = float(hist.iloc[-1]["Low"])
            current_high = float(hist.iloc[-1]["High"])
            current_close = float(hist.iloc[-1]["Close"])
            
            breached = False
            trigger_price = current_close
            if direction == "LONG":
                if current_low <= sl:
                    breached = True
                    trigger_price = min(sl, current_close)
            else: # SHORT
                if current_high >= sl:
                    breached = True
                    trigger_price = max(sl, current_close)
                    
            if breached:
                hit_paper_trade_stop(trade["id"], trigger_price)
                triggered_count += 1
        except Exception as e:
            print(f"[Stop Loss Cron] Error checking {ticker}: {e}", flush=True)
    return triggered_count


def _price_n_days_later(symbol: str, entry_date_str: str, n_trading_days: int) -> float | None:
    """Get the close price N trading days after entry."""
    try:
        entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        # Move N trading days forward
        target = entry
        for _ in range(n_trading_days):
            target = next_trading_day(target)

        # Fetch a window around the target date
        start = (target - timedelta(days=3)).strftime("%Y-%m-%d")
        end = (target + timedelta(days=1)).strftime("%Y-%m-%d")
        t = yf.Ticker(symbol)
        hist = t.history(start=start, end=end)
        if hist.empty:
            return None

        # Find the close on or before target
        for idx in reversed(hist.index):
            if idx.date() <= target:
                return round(float(hist.loc[idx, "Close"]), 2)
        return None
    except Exception:
        return None


def refresh_paper_trade_prices(trade_id: int = None) -> dict:
    """Refresh prices for all active paper trades (or one specific)."""
    trades = list_paper_trades(status="active")
    if trade_id is not None:
        trades = [t for t in trades if t["id"] == trade_id]

    updated_count = 0
    for trade in trades:
        symbol = normalize_ticker(trade["ticker"])
        entry_date = trade["entry_date"]

        # Calculate trading days elapsed
        try:
            entry = datetime.strptime(entry_date, "%Y-%m-%d").date()
            today = date.today()
            trading_days_elapsed = count_trading_days(entry, today)
        except Exception:
            continue

        prices = {}
        # Only fetch prices for horizons that have elapsed
        for horizon_label, days in [("1d", 1), ("3d", 3), ("5d", 5), ("10d", 10)]:
            if trading_days_elapsed >= days:
                existing = trade.get(f"price_{horizon_label}")
                if not existing:
                    price = _price_n_days_later(symbol, entry_date, days)
                    if price:
                        prices[f"price_{horizon_label}"] = price

        # Fetch current price for marking to market
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="1d")
            if not hist.empty:
                current_price = float(hist.iloc[-1]["Close"])
                entry_price = trade["entry_price"]
                direction = trade.get("direction", "LONG")
                multiplier = 1 if direction == "LONG" else -1
                prices["unrealized_pnl_pct"] = round(multiplier * (current_price - entry_price) / entry_price * 100, 2)
        except Exception:
            pass

        if prices:
            update_paper_trade_prices(trade["id"], prices)
            updated_count += 1

        # Auto-expire after 10 trading days
        if trading_days_elapsed > 10 and trade["status"] == "active":
            update_paper_trade_status(trade["id"], "expired")

    # Also refresh shadow trades so they stay in sync with paper trades.
    # Best-effort, never raises.
    shadow_result = None
    try:
        from backend.shadow_trades import refresh_shadow_prices
        shadow_result = refresh_shadow_prices()
    except Exception as e:
        print(f"[Simulation] shadow refresh failed: {e}", flush=True)

    return {
        "ok": True,
        "updated": updated_count,
        "total_active": len(trades),
        "shadow": shadow_result,
    }


def paper_trading_stats() -> dict:
    """Aggregate stats across all paper trades."""
    trades = list_paper_trades()

    def compute_stats(horizon: str):
        key = f"pnl_{horizon}_pct"
        valid = [t for t in trades if t.get(key) is not None]
        if not valid:
            return {"count": 0, "win_rate": 0, "avg_return": 0, "best": 0, "worst": 0}
        wins = sum(1 for t in valid if t[key] > 0)
        total_return = sum(t[key] for t in valid)
        return {
            "count": len(valid),
            "win_rate": round(wins / len(valid) * 100, 1),
            "avg_return": round(total_return / len(valid), 2),
            "best": round(max(t[key] for t in valid), 2),
            "worst": round(min(t[key] for t in valid), 2),
        }

    return {
        "total_trades": len(trades),
        "active": sum(1 for t in trades if t["status"] == "active"),
        "expired": sum(1 for t in trades if t["status"] == "expired"),
        "horizon_1d": compute_stats("1d"),
        "horizon_3d": compute_stats("3d"),
        "horizon_5d": compute_stats("5d"),
        "horizon_10d": compute_stats("10d"),
    }


# ============================================================
# HISTORICAL BACKTEST — run recommender on past dates
# ============================================================

def _compute_rsi(closes, period=14):
    """Calculate Wilder's RSI using exponential smoothing."""
    if len(closes) <= period:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # First value is the simple average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _analyze_stock_at_date(ticker: str, target_date: date, regime: str | None = None) -> dict | None:
    """Replay the recommender logic for a stock AS OF a specific past date.

    Uses historical prices up to (but not including) target_date to compute signals,
    then checks actual prices AFTER target_date to see if the signal worked.
    """
    import numpy as np
    try:
        # Determine regime if not provided
        if regime is None:
            try:
                from backend.market_regime import get_cached_regime
                regime = get_cached_regime(target_date).get("regime")
            except Exception:
                regime = None

        # Fetch active tuned/regime-specific weights
        try:
            from backend.signal_performance import get_active_weights_for_regime
            W = get_active_weights_for_regime(regime)
        except Exception:
            from backend.recommender import DEFAULT_WEIGHTS
            W = DEFAULT_WEIGHTS

        symbol = f"{ticker}.NS"
        t = yf.Ticker(symbol)
        # Fetch data: 6 months BEFORE target + 15 days AFTER for outcome measurement
        start = (target_date - timedelta(days=200)).strftime("%Y-%m-%d")
        end = (target_date + timedelta(days=20)).strftime("%Y-%m-%d")
        hist = t.history(start=start, end=end)
        if hist.empty or len(hist) < 50:
            return None

        # Find the target date index
        target_str = target_date.strftime("%Y-%m-%d")
        idx_options = hist.index[hist.index.strftime("%Y-%m-%d") == target_str]
        if len(idx_options) == 0:
            return None
        target_idx_pos = hist.index.get_loc(idx_options[0])

        # Data AT target (for signals)
        past_hist = hist.iloc[:target_idx_pos + 1]
        if len(past_hist) < 30:
            return None

        closes = past_hist["Close"].values
        highs = past_hist["High"].values
        lows = past_hist["Low"].values
        volumes = past_hist["Volume"].values

        current_close = float(closes[-1])
        prev_close = float(closes[-2])
        current_open = float(past_hist.iloc[-1]["Open"])
        current_high = float(highs[-1])
        current_low = float(lows[-1])
        current_volume = float(volumes[-1])
        avg_volume = float(np.mean(volumes[-20:-1])) if len(volumes) > 20 else current_volume

        # === SIGNAL COMPUTATION (mirrors recommender.py weights) ===
        score = 0.0
        # Gap
        gap_pct = (current_open - prev_close) / prev_close * 100 if prev_close else 0
        if abs(gap_pct) >= 2.0:
            if gap_pct > 0:
                score += W["gap_up_filled"] if current_low <= prev_close else W["gap_up_open"]
            else:
                score += W["gap_down_filled"] if current_high >= prev_close else W["gap_down_open"]

        # Volume + Breakout
        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        if len(highs) > 20:
            n_day_high = float(np.max(highs[-21:-1]))
            n_day_low = float(np.min(lows[-21:-1]))
            if current_high > n_day_high:
                score += W["breakout_vol_confirmed"] if vol_ratio >= 1.5 else W["breakout_weak"]
            elif current_low < n_day_low:
                score += W["breakdown_support"]
        if vol_ratio >= 2.0:
            price_change = (current_close - prev_close) / prev_close * 100 if prev_close else 0
            score += W["volume_bullish"] if price_change > 0.5 else (W["volume_bearish"] if price_change < -0.5 else 0)

        # Support/Resistance
        if len(highs) > 60:
            recent_high = float(np.max(highs[-60:]))
            recent_low = float(np.min(lows[-60:]))
            if (current_close - recent_low) / current_close * 100 < 2.0:
                score += W["near_support"]
            elif (recent_high - current_close) / current_close * 100 < 2.0:
                score += W["near_resistance"]

        # RSI
        rsi = _compute_rsi(closes)
        if rsi is not None:
            if rsi < 30:
                score += W["rsi_oversold"]
            elif rsi > 70:
                score += W["rsi_overbought"]

        # Determine signal + direction
        if score >= 4.0:
            signal = "STRONG BUY"
            direction = 1
        elif score >= 2.0:
            signal = "BUY"
            direction = 1
        elif score <= -4.0:
            signal = "STRONG SELL"
            direction = -1
        elif score <= -2.0:
            signal = "SELL"
            direction = -1
        else:
            signal = "HOLD"
            direction = 1  # For neutral signals, measure price change of holding long

        # === OUTCOME MEASUREMENT ===
        future_hist = hist.iloc[target_idx_pos + 1:]
        if future_hist.empty:
            return None

        returns = {}
        for days in [1, 3, 5, 10]:
            if len(future_hist) >= days:
                future_close = float(future_hist.iloc[days - 1]["Close"])
                ret = direction * (future_close - current_close) / current_close * 100
                returns[f"return_{days}d"] = round(ret, 2)

        return {
            "ticker": ticker,
            "signal": signal,
            "score": round(score, 2),
            "entry_price": round(current_close, 2),
            **returns,
            "outcome_1d": ("win" if returns.get("return_1d", 0) > 0 else "loss") if "return_1d" in returns else None,
            "outcome_5d": ("win" if returns.get("return_5d", 0) > 0 else "loss") if "return_5d" in returns else None,
            "confidence": "HIGH" if abs(score) >= 4 else ("MEDIUM" if abs(score) >= 2.5 else "LOW"),
            "success_probability": None,
        }
    except Exception:
        return None


def run_recommender_backtest(
    universe: str = "nifty50",
    start_date: str = None,
    end_date: str = None,
    interval_days: int = 5,
) -> dict:
    """Run the recommendation engine on historical dates and measure actual outcomes.

    FREE — no AI API cost, pure price math.
    """
    stocks = UNIVERSES.get(universe, [])
    run_id = str(uuid.uuid4())[:8]

    # Default to last 60 days if not specified
    if not end_date:
        end = date.today() - timedelta(days=15)  # Leave room for 10-day outcome
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if not start_date:
        start = end - timedelta(days=60)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Generate trading dates
    dates = []
    current = start
    while current <= end:
        if is_trading_day(current):
            dates.append(current)
            current += timedelta(days=interval_days)
        else:
            current += timedelta(days=1)

    all_results = []

    for d in dates:
        try:
            from backend.market_regime import get_cached_regime
            regime = get_cached_regime(d).get("regime")
        except Exception:
            regime = None

        # Analyze all stocks for this date in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_analyze_stock_at_date, ticker, d, regime) for ticker in stocks]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    result["run_id"] = run_id
                    result["trade_date"] = d.strftime("%Y-%m-%d")
                    save_recommender_backtest_row(result)
                    all_results.append(result)

    # Compute summary stats
    if all_results:
        active_results = [r for r in all_results if r.get("signal") != "HOLD"]
        if active_results:
            wins_5d = sum(1 for r in active_results if r.get("outcome_5d") == "win")
            losses_5d = sum(1 for r in active_results if r.get("outcome_5d") == "loss")
            with_5d = [r for r in active_results if r.get("return_5d") is not None]
            avg_return_5d = sum(r["return_5d"] for r in with_5d) / len(with_5d) if with_5d else 0
        else:
            wins_5d = losses_5d = 0
            avg_return_5d = 0

        # By signal type
        by_signal = {}
        for sig in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            sig_results = [r for r in all_results if r.get("signal") == sig and r.get("return_5d") is not None]
            if sig_results:
                sig_wins = sum(1 for r in sig_results if r["return_5d"] > 0)
                by_signal[sig] = {
                    "count": len(sig_results),
                    "win_rate": round(sig_wins / len(sig_results) * 100, 1),
                    "avg_return": round(sum(r["return_5d"] for r in sig_results) / len(sig_results), 2),
                }
    else:
        wins_5d = losses_5d = 0
        avg_return_5d = 0
        by_signal = {}

    return {
        "run_id": run_id,
        "universe": universe,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "dates_tested": len(dates),
        "total_signals": len(all_results),
        "wins_5d": wins_5d,
        "losses_5d": losses_5d,
        "win_rate_5d": round(wins_5d / (wins_5d + losses_5d) * 100, 1) if (wins_5d + losses_5d) > 0 else 0,
        "avg_return_5d": round(avg_return_5d, 2),
        "by_signal": by_signal,
    }
