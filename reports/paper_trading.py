"""Paper trading log — backtest ↔ live tutarlılığını test etmek için.

Akış:
1) Her gün predict komutu çağrıldığında, shortlist diske yazılır
   (data/processed/paper_trades.parquet).
2) score_outcomes() her gün çağrıldığında, 5/10/20 gün öncesinin
   shortlist'ini alıp gerçek getiriyi hesaplar.
3) Birkaç hafta sonra: paper Sharpe ↔ backtest Sharpe karşılaştır.
   Drift varsa model live'da farklı davranıyor demektir → kök analiz.

Bu mekanizma "in-sample/out-of-sample" disiplinin operasyonel
karşılığı. Backtest'in canlıda kanıtlanması için ZORUNLU.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config import ROOT
from strategy.shortlist import ShortlistItem

PAPER_LOG = ROOT / "data" / "processed" / "paper_trades.parquet"
PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)


def log_predictions(items: list[ShortlistItem], as_of: pd.Timestamp | None = None) -> Path:
    """Bugünkü shortlist'i paper_trades.parquet'a append.

    Her kayıt: prediction_date, ticker, rank, score, entry_price.
    Outcome alanları (5d_ret, 10d_ret, 20d_ret) sonradan doldurulur.
    """
    as_of = as_of or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows = [{
        "prediction_date": as_of,
        "ticker": it.ticker,
        "rank": it.rank,
        "score": it.score,
        "entry_price": it.last_close,
        "ret_5d": None,
        "ret_10d": None,
        "ret_20d": None,
        "outcome_scored_at": None,
    } for it in items]
    new = pd.DataFrame(rows)

    if PAPER_LOG.exists():
        old = pd.read_parquet(PAPER_LOG)
        # Aynı gün için tekrar log atılırsa eskileri sil (idempotent)
        old = old[~((old["prediction_date"] == as_of))]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new

    df.to_parquet(PAPER_LOG, index=False)
    logger.info(f"Paper trades logged: {len(new)} kayıt → {PAPER_LOG.name} (toplam {len(df)})")
    return PAPER_LOG


def score_outcomes(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Geçmiş predictions için 5/10/20 gün sonraki gerçek return'leri doldur.

    Yahoo'dan güncel fiyatları çeker, prediction_date+N gün sonrasının
    close'unu bulur, return hesaplar.
    """
    if not PAPER_LOG.exists():
        logger.warning("Henüz paper trade yok")
        return pd.DataFrame()
    df = pd.read_parquet(PAPER_LOG)
    as_of = as_of or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    # Hangi prediction'lar henüz scoring bekliyor?
    for horizon_days, col in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
        cutoff = as_of - pd.Timedelta(days=horizon_days)
        pending = df[(df[col].isna()) & (df["prediction_date"] <= cutoff)]
        if pending.empty:
            continue
        for idx, row in pending.iterrows():
            outcome = _fetch_close_at(row["ticker"], row["prediction_date"] + pd.Timedelta(days=horizon_days))
            if outcome is not None and row["entry_price"] > 0:
                ret = (outcome / row["entry_price"]) - 1.0
                df.at[idx, col] = round(ret, 5)
                df.at[idx, "outcome_scored_at"] = as_of

    df.to_parquet(PAPER_LOG, index=False)
    logger.info(f"Outcome scoring tamamlandı, {len(df)} toplam paper trade")
    return df


def _fetch_close_at(ticker: str, target_date: pd.Timestamp) -> float | None:
    """Yahoo'dan ticker'ın target_date civarındaki close'unu bul.

    Tatil/hafta sonu için 3 gün ileri ara.
    """
    from features.loader import _safe_path
    p = _safe_path(ticker)
    if not p.exists():
        return None
    px = pd.read_parquet(p)
    px["date"] = pd.to_datetime(px["date"])
    window = px[(px["date"] >= target_date) & (px["date"] <= target_date + pd.Timedelta(days=4))]
    if window.empty:
        return None
    return float(window.iloc[0]["close"])


def summary() -> dict:
    """Paper trading'in özet istatistiği — backtest sayıları ile karşılaştırılacak."""
    if not PAPER_LOG.exists():
        return {"status": "no_data"}
    df = pd.read_parquet(PAPER_LOG)
    out = {
        "total_predictions": int(len(df)),
        "unique_dates": int(df["prediction_date"].nunique()),
        "scored_5d":  int(df["ret_5d"].notna().sum()),
        "scored_10d": int(df["ret_10d"].notna().sum()),
        "scored_20d": int(df["ret_20d"].notna().sum()),
    }
    for col in ("ret_5d", "ret_10d", "ret_20d"):
        s = df[col].dropna()
        if len(s) > 0:
            out[f"{col}_mean"] = round(float(s.mean()), 5)
            out[f"{col}_std"]  = round(float(s.std()), 5)
            out[f"{col}_hit"]  = round(float((s > 0).mean()), 3)
    return out


if __name__ == "__main__":
    print(summary())
