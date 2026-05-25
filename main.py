"""Orchestrator — cron entry point.

Komutlar:
    python main.py ingest    # tüm veri kaynaklarını topla (cron 18:30)
    python main.py score     # haberleri Claude ile skorla (cron 19:00)
    python main.py predict   # günlük top-N shortlist üret (cron 19:30, Faz 3)
    python main.py backtest  # geçmiş veriyle walk-forward (manuel)
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from config import SETTINGS  # noqa: F401  (logger setup)


def cmd_ingest() -> None:
    from ingestion import price_collector, news_collector
    logger.info("=== INGEST: prices ===")
    price_collector.fetch_all()
    logger.info("=== INGEST: news ===")
    news_collector.fetch_all()


def cmd_score() -> None:
    from datetime import datetime
    from pathlib import Path

    import pandas as pd

    from config import ROOT
    from nlp.sentiment_claude import score_news

    today = datetime.utcnow().date().isoformat()
    raw = ROOT / "data" / "raw" / "news" / f"news_{today}.parquet"
    if not raw.exists():
        logger.warning(f"Bugünün ham haberi yok: {raw}")
        return
    df = pd.read_parquet(raw)
    if df.empty:
        logger.info("Bugün haber yok.")
        return
    scored = score_news(df)
    out = ROOT / "data" / "processed" / f"sentiment_{today}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out, index=False)
    logger.info(f"Sentiment yazıldı → {out.name}")


def cmd_predict() -> None:
    from strategy.shortlist import generate_shortlist
    from reports.console import print_shortlist
    from reports.paper_trading import log_predictions, score_outcomes, summary
    items = generate_shortlist()
    print_shortlist(items)
    # Paper trading: predictions log + geçmişlerin outcome'larını skorla
    log_predictions(items)
    score_outcomes()
    print("Paper trading özet:", summary())


def cmd_backtest() -> None:
    from backtest.engine import run_backtest, print_report
    result = run_backtest()
    print_report(result)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ingest", "score", "predict", "backtest"])
    args = p.parse_args()

    cmds = {
        "ingest": cmd_ingest,
        "score": cmd_score,
        "predict": cmd_predict,
        "backtest": cmd_backtest,
    }
    try:
        cmds[args.cmd]()
        return 0
    except KeyboardInterrupt:
        logger.warning("Kullanıcı iptal etti.")
        return 130
    except Exception as e:
        logger.exception(f"Hata: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
