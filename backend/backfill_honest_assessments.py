"""Data Migration: Backfill / recalculate honest assessments for all historical recommendations."""

import json
import sqlite3
import os
import sys

# Add parent directory to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import get_db
from backend.honest_assessment import get_honest_assessment


def backfill_honest_assessments():
    print("Starting data backfill for honest assessments...", flush=True)

    # 1. Backfill paper_trades
    print("\nProcessing paper_trades...", flush=True)
    paper_updated = 0
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, ticker, triggered_signals, score, regime_at_entry FROM paper_trades"
        ).fetchall()

        for r in rows:
            trade_id = r["id"]
            ticker = r["ticker"]
            triggered = r["triggered_signals"]
            score = r["score"] or 0.0
            regime = r["regime_at_entry"]

            # Parse signals
            signals = []
            if triggered:
                try:
                    signals = json.loads(triggered)
                except Exception:
                    signals = []

            # Compute assessment
            assessment = get_honest_assessment(signals, score, regime)
            prob = assessment.get("probability")  # None or int

            conn.execute(
                "UPDATE paper_trades SET success_probability = ? WHERE id = ?",
                (prob, trade_id),
            )
            paper_updated += 1

    print(f"Updated {paper_updated} rows in paper_trades.", flush=True)

    # 2. Backfill shadow_trades
    print("\nProcessing shadow_trades...", flush=True)
    shadow_updated = 0
    with get_db() as conn:
        rows = conn.execute(
            "SELECT ticker, signal_date, triggered_signals, score, regime_at_entry FROM shadow_trades"
        ).fetchall()

        for r in rows:
            ticker = r["ticker"]
            signal_date = r["signal_date"]
            triggered = r["triggered_signals"]
            score = r["score"] or 0.0
            regime = r["regime_at_entry"]

            # Parse signals
            signals = []
            if triggered:
                try:
                    signals = json.loads(triggered)
                except Exception:
                    signals = []

            # Compute assessment
            assessment = get_honest_assessment(signals, score, regime)
            prob = assessment.get("probability")  # None or int

            conn.execute(
                "UPDATE shadow_trades SET success_probability = ? WHERE ticker = ? AND signal_date = ?",
                (prob, ticker, signal_date),
            )
            shadow_updated += 1

    print(f"Updated {shadow_updated} rows in shadow_trades.", flush=True)
    print("\nBackfill migration completed successfully!", flush=True)


if __name__ == "__main__":
    backfill_honest_assessments()
