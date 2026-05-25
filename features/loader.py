"""Per-ticker parquet'leri tek long-format panele dönüştür.

Çıktı şeması:
    date | ticker | open | high | low | close | volume | benchmark_close
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS

PRICE_DIR = ROOT / "data" / "raw" / "prices"


def _safe_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "INDEX_")
    return PRICE_DIR / f"{safe}.parquet"


def load_panel(
    tickers: list[str] | None = None,
    benchmark: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Tüm ticker'ların price verisini long-format panele birleştir.

    Benchmark close kolon olarak panele eklenir (excess return için).
    """
    tickers = tickers or SETTINGS["universe"]["tickers"]
    benchmark = benchmark or SETTINGS["universe"]["benchmark"]

    # Benchmark
    bm_path = _safe_path(benchmark)
    if not bm_path.exists():
        raise FileNotFoundError(f"Benchmark verisi yok: {bm_path}")
    bm = pd.read_parquet(bm_path)[["date", "close"]].rename(columns={"close": "benchmark_close"})
    bm["date"] = pd.to_datetime(bm["date"])

    panels = []
    for t in tickers:
        p = _safe_path(t)
        if not p.exists():
            logger.warning(f"{t}: parquet yok, atlanıyor")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        panels.append(df)

    if not panels:
        raise RuntimeError("Hiç ticker yüklenemedi")

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.merge(bm, on="date", how="left")

    if start is not None:
        panel = panel[panel["date"] >= pd.to_datetime(start)]
    if end is not None:
        panel = panel[panel["date"] <= pd.to_datetime(end)]

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(f"Panel yüklendi: {len(panel):,} satır × {panel['ticker'].nunique()} ticker · "
                f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    return panel
