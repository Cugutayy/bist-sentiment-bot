"""Trend sistemi için fiyat verisi — yfinance, cache'li."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import ROOT

CACHE_DIR = ROOT / "data" / "raw" / "trend"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_prices(universe: list[str], period: str = "10y", use_cache: bool = True) -> pd.DataFrame:
    """Universe için adjusted close paneli (date × ticker).

    Cache: data/raw/trend/prices_{period}.parquet (1 gün taze sayılır).
    """
    cache = CACHE_DIR / f"prices_{period}.parquet"
    if use_cache and cache.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).days
        if age_days < 1:
            df = pd.read_parquet(cache)
            have = [c for c in universe if c in df.columns]
            if len(have) >= len(universe) * 0.8:
                return df[have]

    raw = yf.download(universe, period=period, interval="1d", auto_adjust=True, progress=False)
    close = raw["Close"]
    close = close.dropna(axis=1, thresh=int(len(close) * 0.4))
    close.to_parquet(cache)
    return close
