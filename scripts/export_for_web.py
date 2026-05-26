"""Backtest artifact'larını JSON olarak export — portfolio-tracker site için.

Cloudflare Pages (static) kendi başına Python çalıştıramaz, ama
JSON fetch + JS render edebilir. Bu script veriyi portfolio-tracker
sitesinin tracker/sentiment-bot/data/ klasörüne JSON olarak yazar.

Kullanım:
    python scripts/export_for_web.py
    # Sonra portfolio-tracker repo'sunda git add + commit + push
    # Cloudflare Pages auto-deploy → site güncel
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT

# Portfolio-tracker repo path — env var > yan klasör (Windows) > parent (CI)
_default_windows = Path("C:/Users/cugut/portfolio-tracker")
_default_ci = Path(__file__).resolve().parent.parent.parent / "portfolio-tracker"
if os.environ.get("PORTFOLIO_TRACKER_PATH"):
    TRACKER_REPO = Path(os.environ["PORTFOLIO_TRACKER_PATH"])
elif _default_windows.exists():
    TRACKER_REPO = _default_windows
else:
    TRACKER_REPO = _default_ci
OUT_DIR = TRACKER_REPO / "tracker" / "sentiment-bot" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[export] Hedef: {OUT_DIR}")


def _json_safe(obj):
    """numpy / pandas types → JSON-serializable."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def export_latest_backtest():
    bt_dir = ROOT / "data" / "processed" / "backtests"
    runs = sorted(bt_dir.glob("*/"), reverse=True)
    if not runs:
        print("Backtest yok")
        return
    run = runs[0]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(run / "daily.parquet").set_index("date")
    weights = pd.read_parquet(run / "weights.parquet").set_index("date")

    # Equity curve (date, portfolio, benchmark)
    equity = []
    for d, row in daily.iterrows():
        equity.append({
            "date": d.strftime("%Y-%m-%d"),
            "portfolio": round(float(row["portfolio_equity"]), 4),
            "benchmark": round(float(row["benchmark_equity"]), 4),
            "ret_p": round(float(row["portfolio_return"]), 5),
            "ret_b": round(float(row["benchmark_return"]), 5),
        })

    out = {
        "run_id": summary["run_id"],
        "n_folds": summary["n_folds"],
        "n_days": summary["n_days"],
        "portfolio_metrics": summary["portfolio_metrics"],
        "benchmark_metrics": summary["benchmark_metrics"],
        "fold_metrics": summary["fold_metrics"],
        "equity": equity,
        "generated_at": datetime.now().isoformat(),
    }
    (OUT_DIR / "backtest.json").write_text(json.dumps(out, default=_json_safe, indent=2), encoding="utf-8")
    print(f"  ✓ backtest.json ({len(equity)} gün)")


def export_trades():
    """Public trade ledger — DATA PRIVACY uygulandı.

    YAYINLANANLAR (yatırımcı görür):
      - tarih, ticker, aksiyon (BUY/SELL)
      - pozisyon boyutu (% portföy)
      - entry price, close price
      - günlük P&L (close vs entry)

    SAKLANANLAR (kurumsal IP):
      - Model skoru (predict_proba)
      - Feature rationale (rsi_14, momentum, vs.)
      - Day low/high (gözlemsel kanıt değil, sadece "open'dan giriş" disclosure)

    Bu = rakip kopyalayamaz, sadece performance görür.
    """
    from backtest.trade_explainer import build_trade_ledger
    df = build_trade_ledger()
    if df.empty:
        print("  Trade yok")
        return
    trades = []
    for _, r in df.iterrows():
        # P&L (intraday): entry open → close kapanış
        entry = float(r["entry_price"]) if pd.notna(r["entry_price"]) else None
        close = float(r["close_price"]) if pd.notna(r["close_price"]) else None
        pnl_pct = None
        if entry and close and entry > 0:
            pnl_pct = (close / entry - 1) if r["action"] == "BUY" else (entry / close - 1)
            pnl_pct = round(pnl_pct, 5)
        trades.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "ticker": r["ticker"],
            "action": r["action"],
            "size_pct": round(float(r["weight_delta"]) * 100, 2),     # pozisyon %
            "entry_price": round(entry, 2) if entry else None,
            "close_price": round(close, 2) if close else None,
            "pnl_pct": pnl_pct,                                         # günlük kâr/zarar %
            # SAKLANAN: score, rationale, day_low, day_high
        })
    (OUT_DIR / "trades.json").write_text(json.dumps(trades, indent=2), encoding="utf-8")
    print(f"  ✓ trades.json ({len(trades)} trade) — public fields only, IP korundu")


def export_shortlist():
    """Public shortlist — sadece rank + ticker + fiyat. Skor/features SAKLI."""
    proc = ROOT / "data" / "processed"
    files = sorted(proc.glob("shortlist_*.parquet"), reverse=True)
    if not files:
        return
    df = pd.read_parquet(files[0])
    items = []
    for i, (_, r) in enumerate(df.iterrows()):
        items.append({
            "rank": i + 1,
            "ticker": r["ticker"],
            "close": round(float(r.get("close", 0)), 2),
            # SAKLANAN: score, ret_5, momentum_60, rsi_14
        })
    out = {
        "date": files[0].stem.replace("shortlist_", ""),
        "items": items,
        "note": "Sıralama proprietary model çıktısıdır. Feature & skor detayı kurumsal IP.",
    }
    (OUT_DIR / "shortlist.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  ✓ shortlist.json ({len(items)} ticker) — rank-only public view")


def export_paper_log():
    paper = ROOT / "data" / "processed" / "paper_trades.parquet"
    if not paper.exists():
        return
    df = pd.read_parquet(paper)
    items = []
    for _, r in df.iterrows():
        items.append({
            "prediction_date": pd.to_datetime(r["prediction_date"]).strftime("%Y-%m-%d"),
            "ticker": r["ticker"],
            "rank": int(r["rank"]),
            "score": round(float(r["score"]), 3),
            "entry_price": round(float(r["entry_price"]), 2),
            "ret_5d": round(float(r["ret_5d"]), 4) if pd.notna(r["ret_5d"]) else None,
            "ret_10d": round(float(r["ret_10d"]), 4) if pd.notna(r["ret_10d"]) else None,
            "ret_20d": round(float(r["ret_20d"]), 4) if pd.notna(r["ret_20d"]) else None,
        })
    (OUT_DIR / "paper.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"  ✓ paper.json ({len(items)} kayıt)")


def export_paper_book():
    """Açık/kapalı pozisyon book'u — investor-facing (skor saklı)."""
    book_path = ROOT / "data" / "processed" / "paper_book.parquet"
    if not book_path.exists():
        print("  Paper book yok (henüz pozisyon açılmadı)")
        return
    df = pd.read_parquet(book_path)
    if df.empty:
        return

    # Investor-safe: skor/rationale gizli; sadece zaman/fiyat/PnL public
    trades = []
    for _, r in df.iterrows():
        trades.append({
            "trade_id": r["trade_id"],
            "open_date": pd.to_datetime(r["prediction_date"]).strftime("%Y-%m-%d"),
            "ticker": r["ticker"],
            "entry_price": round(float(r["entry_price"]), 2),
            "status": r["status"],   # OPEN / CLOSED
            "close_date": pd.to_datetime(r["close_date"]).strftime("%Y-%m-%d") if pd.notna(r["close_date"]) else None,
            "close_price": round(float(r["close_price"]), 2) if pd.notna(r["close_price"]) else None,
            "close_reason": r["close_reason"] if pd.notna(r["close_reason"]) else None,
            "pnl_pct": round(float(r["pnl_pct"]), 5) if pd.notna(r["pnl_pct"]) else None,
            "days_held": int(r["days_held_at_close"]) if pd.notna(r["days_held_at_close"]) else None,
            "tp_pct": round(float(r["tp_pct"]), 4),
            "sl_pct": round(float(r["sl_pct"]), 4),
            "max_hold_days": int(r["max_hold_days"]),
        })

    # Özet
    closed = [t for t in trades if t["status"] == "CLOSED"]
    open_ = [t for t in trades if t["status"] == "OPEN"]
    pnls = [t["pnl_pct"] for t in closed if t["pnl_pct"] is not None]
    summary = {
        "total": len(trades),
        "open": len(open_),
        "closed": len(closed),
        "closed_tp": sum(1 for t in closed if t["close_reason"] == "TP"),
        "closed_sl": sum(1 for t in closed if t["close_reason"] == "SL"),
        "closed_time": sum(1 for t in closed if t["close_reason"] == "TIME"),
        "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else None,
        "avg_pnl": round(sum(pnls) / len(pnls), 5) if pnls else None,
        "total_pnl": round(sum(pnls), 5) if pnls else None,
        "best": round(max(pnls), 5) if pnls else None,
        "worst": round(min(pnls), 5) if pnls else None,
    }

    out = {
        "trades": trades,
        "summary": summary,
        "generated_at": datetime.now().isoformat(),
    }
    (OUT_DIR / "paper_book.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  ✓ paper_book.json ({len(trades)} trade · {summary['open']} açık · {summary['closed']} kapalı)")


if __name__ == "__main__":
    print(f"=== Export → {OUT_DIR} ===")
    export_latest_backtest()
    export_trades()
    export_shortlist()
    export_paper_log()
    export_paper_book()
    print("Bitti. Portfolio-tracker repo'sunda commit + push gerekli.")
