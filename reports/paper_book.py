"""Paper trading book — open/closed pozisyon yönetimi.

Backtest'in birebir aynası: triple barrier (TP/SL/zaman) ile pozisyon kapanışı.

Akış:
1) Her gün predict çalışınca: top-N ticker için OPEN pozisyon kaydı
   (aynı ticker zaten açıksa skipper).
2) update_positions(): tüm OPEN pozisyonlar için en güncel kapanışı getir,
   triple barrier check yap (TP / SL / max-hold), kapanan pozisyonları
   CLOSED olarak işaretle ve PnL hesapla.

Veri:
    data/processed/paper_book.parquet
    columns: trade_id, prediction_date, ticker, entry_price, status,
             close_date, close_price, close_reason, pnl_pct,
             tp_pct, sl_pct, max_hold_days, days_held_at_close

close_reason: 'TP' (take-profit), 'SL' (stop-loss), 'TIME' (max-hold dolunca)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS
from strategy.shortlist import ShortlistItem

PAPER_BOOK = ROOT / "data" / "processed" / "paper_book.parquet"
PAPER_BOOK.parent.mkdir(parents=True, exist_ok=True)


def _empty_book() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_id", "prediction_date", "ticker", "entry_price", "status",
        "close_date", "close_price", "close_reason", "pnl_pct",
        "tp_pct", "sl_pct", "max_hold_days", "days_held_at_close",
    ])


def _load_book() -> pd.DataFrame:
    if PAPER_BOOK.exists():
        df = pd.read_parquet(PAPER_BOOK)
        # Ensure date dtypes
        df["prediction_date"] = pd.to_datetime(df["prediction_date"])
        if df["close_date"].notna().any():
            df["close_date"] = pd.to_datetime(df["close_date"])
        return df
    return _empty_book()


def _save_book(df: pd.DataFrame) -> None:
    df.to_parquet(PAPER_BOOK, index=False)


def open_positions(items: list[ShortlistItem], as_of: pd.Timestamp | None = None) -> int:
    """Top-N shortlist'i OPEN pozisyon olarak kaydet.

    Aynı ticker zaten açıksa atla (yenisini açma — double dip önlemi).
    Returns: kaç yeni pozisyon açıldı.
    """
    as_of = as_of or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    labels_cfg = SETTINGS["labels"]
    tp = float(labels_cfg["profit_target_pct"])
    sl = float(labels_cfg["stop_loss_pct"])
    mh = int(labels_cfg["max_holding_days"])

    book = _load_book()
    open_tickers = set(book[book["status"] == "OPEN"]["ticker"].tolist())

    rows = []
    for it in items:
        if it.ticker in open_tickers:
            continue
        rows.append({
            "trade_id": uuid4().hex[:12],
            "prediction_date": as_of,
            "ticker": it.ticker,
            "entry_price": float(it.last_close),
            "status": "OPEN",
            "close_date": pd.NaT,
            "close_price": None,
            "close_reason": None,
            "pnl_pct": None,
            "tp_pct": tp,
            "sl_pct": sl,
            "max_hold_days": mh,
            "days_held_at_close": None,
        })

    if not rows:
        logger.info(f"Paper book: yeni pozisyon yok (hepsi zaten açık) — {as_of.date()}")
        return 0

    new_df = pd.DataFrame(rows)
    book = pd.concat([book, new_df], ignore_index=True) if not book.empty else new_df
    _save_book(book)
    logger.info(f"Paper book: {len(rows)} yeni OPEN pozisyon @ {as_of.date()} (toplam açık: {(book['status']=='OPEN').sum()})")
    return len(rows)


def _fetch_price_path(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Bir ticker için [start, end] arası günlük OHLC'yi diskten oku."""
    from features.loader import _safe_path
    p = _safe_path(ticker)
    if not p.exists():
        return pd.DataFrame()
    px = pd.read_parquet(p)
    px["date"] = pd.to_datetime(px["date"])
    return px[(px["date"] >= start) & (px["date"] <= end)].sort_values("date").reset_index(drop=True)


def update_positions(as_of: pd.Timestamp | None = None) -> dict:
    """Tüm OPEN pozisyonlar için triple barrier check yap.

    Her açık trade için:
    - Entry'den as_of'a kadar olan fiyat path'ini çek
    - TP veya SL'a vurduğu ilk günü bul
    - Vurmadıysa, days_held >= max_hold_days mi kontrol et (zaman bariyeri)
    - Vurdu/doldu ise CLOSED işaretle, pnl hesapla

    Returns: {'closed_tp': n, 'closed_sl': n, 'closed_time': n, 'still_open': n}
    """
    as_of = as_of or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    book = _load_book()
    if book.empty:
        logger.warning("Paper book boş, kapanacak pozisyon yok")
        return {"closed_tp": 0, "closed_sl": 0, "closed_time": 0, "still_open": 0}

    open_mask = book["status"] == "OPEN"
    open_idx = book[open_mask].index.tolist()
    stats = {"closed_tp": 0, "closed_sl": 0, "closed_time": 0, "still_open": 0}

    for idx in open_idx:
        row = book.loc[idx]
        entry = float(row["entry_price"])
        if entry <= 0:
            continue
        tp = float(row["tp_pct"])
        sl = float(row["sl_pct"])
        mh = int(row["max_hold_days"])
        pred_date = pd.to_datetime(row["prediction_date"])

        # Fiyat path'i (entry günü dahil değil — entry günü zaten alım fiyatı)
        path = _fetch_price_path(row["ticker"], pred_date + pd.Timedelta(days=1), as_of)
        if path.empty:
            stats["still_open"] += 1
            continue

        closed = False
        for i, prow in path.iterrows():
            day_close = float(prow["close"])
            ret = (day_close / entry) - 1.0
            days_held = (prow["date"] - pred_date).days

            if ret >= tp:
                book.at[idx, "status"] = "CLOSED"
                book.at[idx, "close_date"] = prow["date"]
                book.at[idx, "close_price"] = day_close
                book.at[idx, "close_reason"] = "TP"
                book.at[idx, "pnl_pct"] = round(ret, 5)
                book.at[idx, "days_held_at_close"] = days_held
                stats["closed_tp"] += 1
                closed = True
                break
            if ret <= -sl:
                book.at[idx, "status"] = "CLOSED"
                book.at[idx, "close_date"] = prow["date"]
                book.at[idx, "close_price"] = day_close
                book.at[idx, "close_reason"] = "SL"
                book.at[idx, "pnl_pct"] = round(ret, 5)
                book.at[idx, "days_held_at_close"] = days_held
                stats["closed_sl"] += 1
                closed = True
                break
            if days_held >= mh:
                book.at[idx, "status"] = "CLOSED"
                book.at[idx, "close_date"] = prow["date"]
                book.at[idx, "close_price"] = day_close
                book.at[idx, "close_reason"] = "TIME"
                book.at[idx, "pnl_pct"] = round(ret, 5)
                book.at[idx, "days_held_at_close"] = days_held
                stats["closed_time"] += 1
                closed = True
                break

        if not closed:
            stats["still_open"] += 1

    _save_book(book)
    logger.info(
        f"Paper book update @ {as_of.date()}: "
        f"TP={stats['closed_tp']} SL={stats['closed_sl']} TIME={stats['closed_time']} "
        f"OPEN={stats['still_open']}"
    )
    return stats


def summary() -> dict:
    """Genel paper book özet — investor-facing metrikler."""
    book = _load_book()
    if book.empty:
        return {"status": "no_data"}
    closed = book[book["status"] == "CLOSED"]
    open_ = book[book["status"] == "OPEN"]
    pnls = closed["pnl_pct"].dropna().astype(float)

    out = {
        "total_trades": int(len(book)),
        "open_trades": int(len(open_)),
        "closed_trades": int(len(closed)),
        "closed_tp": int((closed["close_reason"] == "TP").sum()),
        "closed_sl": int((closed["close_reason"] == "SL").sum()),
        "closed_time": int((closed["close_reason"] == "TIME").sum()),
        "win_rate": round(float((pnls > 0).mean()), 4) if len(pnls) else None,
        "avg_pnl_pct": round(float(pnls.mean()), 5) if len(pnls) else None,
        "total_pnl_pct": round(float(pnls.sum()), 5) if len(pnls) else None,
        "best_trade_pct": round(float(pnls.max()), 5) if len(pnls) else None,
        "worst_trade_pct": round(float(pnls.min()), 5) if len(pnls) else None,
        "avg_days_held": round(float(closed["days_held_at_close"].dropna().mean()), 2) if len(closed) else None,
    }
    return out


if __name__ == "__main__":
    print("=== Paper book summary ===")
    s = summary()
    for k, v in s.items():
        print(f"  {k}: {v}")
