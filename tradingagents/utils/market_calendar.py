"""Indian market calendar utilities — NSE/BSE trading hours, holidays, sessions."""

from datetime import date, datetime, time, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_MARKET_OPEN = time(9, 0)
POST_MARKET_CLOSE = time(16, 0)

# NSE holidays for 2025-2026 (update annually)
# Source: NSE circulars
NSE_HOLIDAYS = {
    # 2025
    date(2025, 2, 26),  # Mahashivratri
    date(2025, 3, 14),  # Holi
    date(2025, 3, 31),  # Id-Ul-Fitr (Ramadan)
    date(2025, 4, 10),  # Shri Mahavir Jayanti
    date(2025, 4, 14),  # Dr. Ambedkar Jayanti
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 1),   # Maharashtra Day
    date(2025, 8, 15),  # Independence Day
    date(2025, 8, 27),  # Ganesh Chaturthi
    date(2025, 10, 1),  # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 2),  # Dussehra
    date(2025, 10, 21), # Diwali Laxmi Pujan
    date(2025, 10, 22), # Diwali Balipratipada
    date(2025, 11, 5),  # Guru Nanak Jayanti
    date(2025, 12, 25), # Christmas
    # 2026 (placeholder — update when NSE publishes)
    date(2026, 1, 26),  # Republic Day
    date(2026, 3, 3),   # Mahashivratri (approx)
    date(2026, 3, 30),  # Holi (approx)
    date(2026, 4, 3),   # Good Friday (approx)
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),  # Gandhi Jayanti
    date(2026, 11, 9),  # Diwali (approx)
    date(2026, 12, 25), # Christmas
}


def is_market_open(dt: datetime = None) -> bool:
    """Check if the Indian stock market is currently open."""
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    if not is_trading_day(dt.date()):
        return False

    current_time = dt.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_trading_day(d: date = None) -> bool:
    """Check if a given date is a trading day (weekday + not a holiday)."""
    if d is None:
        d = datetime.now(IST).date()
    # Weekends
    if d.weekday() >= 5:
        return False
    # NSE holidays
    if d in NSE_HOLIDAYS:
        return False
    return True


def next_trading_day(d: date = None) -> date:
    """Get the next trading day after the given date."""
    if d is None:
        d = datetime.now(IST).date()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(d: date = None) -> date:
    """Get the previous trading day before the given date."""
    if d is None:
        d = datetime.now(IST).date()
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def get_market_session(dt: datetime = None) -> str:
    """Get the current market session.

    Returns one of: "pre_market", "open", "closing_hour", "post_market", "closed"
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)

    if not is_trading_day(dt.date()):
        return "closed"

    current_time = dt.time()

    if current_time < PRE_MARKET_OPEN:
        return "closed"
    elif current_time < MARKET_OPEN:
        return "pre_market"
    elif current_time <= time(14, 30):
        return "open"
    elif current_time <= MARKET_CLOSE:
        return "closing_hour"
    elif current_time <= POST_MARKET_CLOSE:
        return "post_market"
    else:
        return "closed"


def count_trading_days(start: date, end: date) -> int:
    """Count trading days between start (exclusive) and end (inclusive)."""
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    if start >= end:
        return 0
    count = 0
    curr = start
    while curr < end:
        curr += timedelta(days=1)
        if is_trading_day(curr):
            count += 1
    return count

