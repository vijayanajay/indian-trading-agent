"""Migration script to backfill fingerprints for existing paper_trades and shadow_trades.

Computes the fingerprint by hashing the sorted list of signal type strings plus the
regime label (regime_at_entry) and writes it back to the database.
"""

from __future__ import annotations

import json
import logging
from backend.db import get_db, _migrate_paper_trades_columns
from backend.honest_assessment import compute_fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill-fingerprints")


def backfill_fingerprints() -> dict:
    """Compute and update signal_fingerprint for existing trades."""
    _migrate_paper_trades_columns()

    paper_scanned = 0
    paper_updated = 0
    shadow_scanned = 0
    shadow_updated = 0

    # 1. Backfill paper_trades
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, triggered_signals, regime_at_entry, signal_fingerprint
               FROM paper_trades"""
        ).fetchall()

    for r in rows:
        paper_scanned += 1
        signals = []
        if r["triggered_signals"]:
            try:
                signals = json.loads(r["triggered_signals"])
            except Exception:
                pass
        if not isinstance(signals, list):
            signals = []

        signal_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]
        regime = r["regime_at_entry"]
        fingerprint = compute_fingerprint(signal_types, regime)

        # Update if it differs
        if fingerprint != r["signal_fingerprint"]:
            with get_db() as conn:
                conn.execute(
                    "UPDATE paper_trades SET signal_fingerprint = ? WHERE id = ?",
                    (fingerprint, r["id"]),
                )
            paper_updated += 1

    # 2. Backfill shadow_trades
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ticker, signal_date, triggered_signals, regime_at_entry, signal_fingerprint
               FROM shadow_trades"""
        ).fetchall()

    for r in rows:
        shadow_scanned += 1
        signals = []
        if r["triggered_signals"]:
            try:
                signals = json.loads(r["triggered_signals"])
            except Exception:
                pass
        if not isinstance(signals, list):
            signals = []

        signal_types = [s.get("type") for s in signals if isinstance(s, dict) and s.get("type")]
        regime = r["regime_at_entry"]
        fingerprint = compute_fingerprint(signal_types, regime)

        # Update if it differs
        if fingerprint != r["signal_fingerprint"]:
            with get_db() as conn:
                conn.execute(
                    "UPDATE shadow_trades SET signal_fingerprint = ? WHERE ticker = ? AND signal_date = ?",
                    (fingerprint, r["ticker"], r["signal_date"]),
                )
            shadow_updated += 1

    logger.info(
        f"Backfill complete: "
        f"paper_trades scanned={paper_scanned}, updated={paper_updated}; "
        f"shadow_trades scanned={shadow_scanned}, updated={shadow_updated}"
    )

    return {
        "paper_scanned": paper_scanned,
        "paper_updated": paper_updated,
        "shadow_scanned": shadow_scanned,
        "shadow_updated": shadow_updated,
    }


if __name__ == "__main__":
    backfill_fingerprints()
