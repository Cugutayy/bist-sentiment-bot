"""BIST hisse fiyatları + XU100 benchmark — Yahoo Finance üzerinden.

Ham veri data/raw/prices/<TICKER>.parquet olarak yazılır.
Her ticker append-only — varsa son tarihten itibaren incremental fetch.

Notlar
-----
- Yahoo BIST verisi bazen geç güncellenir (1-2 gün gecikme normal).
- Retry/backoff için tenacity kullanılır.
- Adjusted close (auto_adjust=True) kullanılır → temettü/split düzeltilmiş.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import ROOT, SETTINGS

RAW_PRICE_DIR = ROOT / "data" / "raw" / "prices"
RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _fetch_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Tek bir ticker için Yahoo'dan günlük OHLCV çek."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo boş veri döndü: {ticker}")
    # Yahoo bazen MultiIndex columns döner (ticker layer); flatten et
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    # yfinance 1.4: index.name None olabilir → ilk kolonu açıkça "date" yap
    df.index.name = "date"
    df = df.reset_index()
    # Bazı sürümlerde reset_index "index" döndürür — onu da yakala
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _output_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "INDEX_")
    return RAW_PRICE_DIR / f"{safe}.parquet"


def fetch_one(ticker: str, force_full: bool = False) -> pd.DataFrame:
    """Bir ticker'ı incremental olarak günceller; varsa eski parquet'a append.

    Parameters
    ----------
    ticker : str
        Yahoo sembolü (örn. "THYAO.IS", "XU100.IS").
    force_full : bool
        True ise eski veri silinip baştan çekilir (5 yıllık).
    """
    path = _output_path(ticker)
    history_days = SETTINGS["ingestion"]["price_history_days"]
    default_start = (datetime.utcnow() - timedelta(days=history_days)).date().isoformat()

    if path.exists() and not force_full:
        existing = pd.read_parquet(path)
        last = existing["date"].max()
        start = (last + timedelta(days=1)).date().isoformat()
        if pd.to_datetime(start) > datetime.utcnow():
            logger.info(f"{ticker}: zaten güncel (son: {last.date()})")
            return existing
        logger.info(f"{ticker}: incremental fetch from {start}")
    else:
        existing = None
        start = default_start
        logger.info(f"{ticker}: full fetch from {start}")

    try:
        new = _fetch_ticker(ticker, start=start)
    except Exception as e:
        logger.warning(f"{ticker}: fetch fail → {e}")
        return existing if existing is not None else pd.DataFrame()

    if existing is not None and not new.empty:
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"]).sort_values("date")
    else:
        combined = new

    combined.to_parquet(path, index=False)
    logger.info(f"{ticker}: {len(combined)} satır kaydedildi → {path.name}")
    return combined


def fetch_all(force_full: bool = False) -> dict[str, pd.DataFrame]:
    """Tüm universe + benchmark için fiyat çek."""
    tickers = SETTINGS["universe"]["tickers"]
    benchmark = SETTINGS["universe"]["benchmark"]
    out: dict[str, pd.DataFrame] = {}

    for t in tickers + [benchmark]:
        df = fetch_one(t, force_full=force_full)
        if not df.empty:
            out[t] = df

    logger.info(f"Price ingestion bitti — {len(out)}/{len(tickers)+1} sembol başarılı.")
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="5 yıllık tam fetch (yavaş)")
    p.add_argument("--ticker", type=str, help="Sadece bir ticker'ı güncelle")
    args = p.parse_args()

    if args.ticker:
        fetch_one(args.ticker, force_full=args.full)
    else:
        fetch_all(force_full=args.full)
