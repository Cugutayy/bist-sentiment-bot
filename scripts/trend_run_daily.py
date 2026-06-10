"""Trend bot canli paper trading - gunluk runner.

Akis:
  1) update_paper_book() ile bugunku islemleri yap
  2) Tum ciktilari portfolio-tracker'in tracker/trend-bot/data/ klasorune kopyala
  3) Commit + push (GitHub Actions yapacaksa skip)

Cron icin GitHub Actions: .github/workflows/trend_daily.yml
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# proje koku
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trend.paper_book import (
    update_paper_book, BOOK_DIR, STATE_PATH, EQUITY_PATH,
    TRADES_PATH, POSITIONS_PATH, METRICS_PATH,
)

# portfolio-tracker tarafindaki hedef klasor
TRACKER_REPO = Path("C:/Users/cugut/portfolio-tracker")
SITE_DATA_DIR = TRACKER_REPO / "tracker" / "trend-bot" / "data"


def export_to_site():
    """Paper book ciktilarini site'a kopyala."""
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [STATE_PATH, EQUITY_PATH, TRADES_PATH, POSITIONS_PATH, METRICS_PATH]
    for src in files:
        if src.exists():
            shutil.copy(src, SITE_DATA_DIR / src.name)
            print(f"  copied {src.name}")


def main():
    print("=" * 70)
    print("TREND BOT — DAILY PAPER TRADING RUN")
    print("=" * 70)

    print("\n[1/2] Updating paper book...")
    result = update_paper_book()
    print(json.dumps(result, indent=2, default=str))

    print("\n[2/2] Exporting to site...")
    export_to_site()

    print("\nDone.")


if __name__ == "__main__":
    main()
