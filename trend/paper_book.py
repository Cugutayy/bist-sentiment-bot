"""Trend strategisi icin paper trading defteri.

Her gun calistirir:
  1) En son fiyatlari yfinance'den ceker (cache)
  2) Dunden kalan pozisyonlari bugunku close ile yeniden degerle
  3) Strateji blended_signal -> hedef agirliklari uretir
  4) Hedef != gercek ise rebalance kaydeder
  5) JSON dosyalarini gunceller (site bunlardan beslenir)

Output dosyalari (data/processed/trend_paper/):
  state.json      - guncel pozisyonlar + son guncelleme tarihi
  equity.json     - gunluk equity curve [{date, value, return, cumulative}]
  trades.json     - rebalance olaylari [{date, ticker, action, weight_old, weight_new, price}]
  positions.json  - guncel acik pozisyonlar [{ticker, weight, qty, entry_price, current_value}]
  metrics.json    - performance metrikleri (cagr, sharpe, dd, vs SPY)
  benchmark.json  - SPY equity curve
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT
from trend.data import fetch_prices
from trend.strategy import (TrendConfig, DEFAULT_UNIVERSE, blended_signal,
                            compute_metrics)


BOOK_DIR = ROOT / "data" / "processed" / "trend_paper"
BOOK_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH      = BOOK_DIR / "state.json"
EQUITY_PATH     = BOOK_DIR / "equity.json"
TRADES_PATH     = BOOK_DIR / "trades.json"
POSITIONS_PATH  = BOOK_DIR / "positions.json"
METRICS_PATH    = BOOK_DIR / "metrics.json"
BENCHMARK_PATH  = BOOK_DIR / "benchmark.json"

INITIAL_CAPITAL = 100_000.0      # USD baslangic
LIVE_START      = "2026-06-09"   # canli baslama tarihi (bugun)
LIVE_END        = "2026-09-09"   # 3 ay sonra


# ─── Data classes ───────────────────────────────────────────────────

@dataclass
class BookState:
    last_update: str
    cash: float
    positions: dict[str, dict]   # ticker -> {weight, qty, entry_price, entry_date}
    total_value: float
    benchmark_value: float
    days_live: int

    @classmethod
    def initial(cls):
        return cls(
            last_update="",
            cash=INITIAL_CAPITAL,
            positions={},
            total_value=INITIAL_CAPITAL,
            benchmark_value=INITIAL_CAPITAL,
            days_live=0,
        )


# ─── Helpers ────────────────────────────────────────────────────────

def _load_state() -> BookState:
    if STATE_PATH.exists():
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return BookState(**d)
    return BookState.initial()


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _append_json(path: Path, record: dict) -> None:
    arr = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    arr.append(record)
    path.write_text(json.dumps(arr, indent=2, default=str), encoding="utf-8")


def _compute_target_weights(close: pd.DataFrame, cfg: TrendConfig) -> dict[str, float]:
    """Sinyali ve pozisyon olcusunu hesaplayip son gunun hedef agirligini doner."""
    rets = close.pct_change()
    sig = blended_signal(close, cfg)
    cvol = rets.rolling(cfg.vol_lookback).std() * np.sqrt(252)
    inv_vol = (1.0 / cvol).clip(0, 50).shift(1)
    pos = sig * inv_vol
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    pos = pos.div(gross, axis=0).fillna(0.0)
    # Vol-target scale (rolling realized)
    raw = (pos * rets).sum(axis=1) - pos.diff().abs().sum(axis=1) * cfg.cost_per_trade
    realized = raw.rolling(60, min_periods=20).std() * np.sqrt(252)
    scale = (cfg.vol_target_annual / realized).shift(1).clip(0, cfg.leverage_cap).fillna(1.0)
    pos = pos.multiply(scale, axis=0).fillna(0.0)
    # Son gun
    last = pos.iloc[-1]
    return {t: float(w) for t, w in last.items() if abs(w) > 0.001}


def _live_dates(close: pd.DataFrame) -> list[pd.Timestamp]:
    """LIVE_START ve sonrasi tarihler."""
    start = pd.Timestamp(LIVE_START)
    return [d for d in close.index if d >= start]


# ─── Main update routine ───────────────────────────────────────────

def update_paper_book(asof: pd.Timestamp | None = None,
                      cfg: TrendConfig | None = None,
                      universe: list[str] | None = None) -> dict:
    """Paper book'i guncelle. asof = bugune kadar olan veriyi kullan.

    Returns: bugun yapilan islemlerin ozeti."""
    cfg = cfg or TrendConfig(vol_target_annual=0.15)
    universe = universe or list(DEFAULT_UNIVERSE)

    # Fiyatlari cek
    close = fetch_prices(universe, period="10y")
    if asof is not None:
        close = close[close.index <= asof]
    state = _load_state()
    today = close.index[-1].strftime("%Y-%m-%d")

    if state.last_update == today:
        return {"status": "already_updated", "date": today}

    summary = {"date": today, "actions": [], "value": 0, "benchmark": 0}

    # 1. Dunden kalan pozisyonlari bugunku close ile degerle
    if state.positions:
        new_total = state.cash
        for ticker, pos in state.positions.items():
            if ticker not in close.columns: continue
            cur_price = float(close[ticker].iloc[-1])
            qty = pos["qty"]
            value = qty * cur_price
            pos["current_price"] = cur_price
            pos["current_value"] = value
            new_total += value
        state.total_value = new_total

    # 2. Benchmark (SPY) buy & hold from LIVE_START
    if "SPY" in close.columns:
        live_dates = _live_dates(close)
        if live_dates:
            spy_series = close["SPY"].reindex(live_dates).dropna()
            if len(spy_series) > 0:
                first_price = float(spy_series.iloc[0])
                last_price = float(spy_series.iloc[-1])
                state.benchmark_value = INITIAL_CAPITAL * (last_price / first_price)

    # 3. Yeni hedef agirliklari hesapla
    target_weights = _compute_target_weights(close, cfg)
    summary["target_weights"] = target_weights

    # 4. Mevcut vs hedef kıyasla - rebalance eden ticker'lar
    REBALANCE_THRESHOLD = 0.05  # %5'lik agirlik degisikligi rebalance tetikler
    current_weights = {
        t: pos["weight"] for t, pos in state.positions.items()
    }
    all_tickers = set(target_weights.keys()) | set(current_weights.keys())

    rebalance_done = False
    for ticker in sorted(all_tickers):
        cur_w = current_weights.get(ticker, 0.0)
        tgt_w = target_weights.get(ticker, 0.0)
        if abs(tgt_w - cur_w) < REBALANCE_THRESHOLD:
            continue
        if ticker not in close.columns: continue
        price = float(close[ticker].iloc[-1])
        target_value = state.total_value * tgt_w
        target_qty = target_value / price if price > 0 else 0
        old_qty = state.positions.get(ticker, {}).get("qty", 0.0)
        action = "BUY" if tgt_w > cur_w else ("SELL" if tgt_w < cur_w else "HOLD")

        if abs(tgt_w) < 0.005:
            # Pozisyonu kapat
            if ticker in state.positions:
                del state.positions[ticker]
            action = "CLOSE"
        else:
            state.positions[ticker] = {
                "ticker": ticker, "weight": tgt_w, "qty": target_qty,
                "entry_price": price, "entry_date": today,
                "current_price": price, "current_value": target_qty * price,
            }

        rebal = {
            "date": today, "ticker": ticker, "action": action,
            "weight_old": cur_w, "weight_new": tgt_w,
            "qty_old": old_qty, "qty_new": target_qty,
            "price": price,
        }
        summary["actions"].append(rebal)
        rebalance_done = True

    # 5. Cash = total - tum pozisyonlar
    total_pos_value = sum(p["qty"] * p["current_price"]
                          for p in state.positions.values())
    state.cash = state.total_value - total_pos_value

    # 6. State + audit logging
    state.last_update = today
    state.days_live = max(0, (pd.Timestamp(today) - pd.Timestamp(LIVE_START)).days)

    _save_json(STATE_PATH, asdict(state))

    # Equity curve append
    equity_record = {
        "date": today,
        "value": round(state.total_value, 2),
        "benchmark": round(state.benchmark_value, 2),
        "return": round((state.total_value / INITIAL_CAPITAL - 1) * 100, 4),
        "benchmark_return": round((state.benchmark_value / INITIAL_CAPITAL - 1) * 100, 4),
        "n_positions": len(state.positions),
    }
    summary["value"] = state.total_value
    summary["benchmark"] = state.benchmark_value
    summary["return"] = equity_record["return"]
    summary["benchmark_return"] = equity_record["benchmark_return"]

    eq_arr = json.loads(EQUITY_PATH.read_text()) if EQUITY_PATH.exists() else []
    # Aynı gunde mukerrer eklemeyelim
    eq_arr = [r for r in eq_arr if r["date"] != today]
    eq_arr.append(equity_record)
    eq_arr.sort(key=lambda r: r["date"])
    _save_json(EQUITY_PATH, eq_arr)

    # Trades dosyasi
    if summary["actions"]:
        for action in summary["actions"]:
            _append_json(TRADES_PATH, action)

    # Positions snapshot
    positions_list = [
        {**p, "ticker": t} for t, p in state.positions.items()
    ]
    _save_json(POSITIONS_PATH, positions_list)

    # Metrics
    _update_metrics(eq_arr, close)

    summary["status"] = "updated"
    summary["n_actions"] = len(summary["actions"])
    return summary


def _update_metrics(equity_arr: list[dict], close: pd.DataFrame) -> None:
    """Performance metriklerini hesapla, JSON'a yaz."""
    if len(equity_arr) < 2: return
    df = pd.DataFrame(equity_arr)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    rets = df["value"].pct_change().dropna()
    bench_rets = df["benchmark"].pct_change().dropna()

    n = len(rets)
    if n < 2: return
    cagr = (df["value"].iloc[-1] / df["value"].iloc[0]) ** (252/n) - 1
    bench_cagr = (df["benchmark"].iloc[-1] / df["benchmark"].iloc[0]) ** (252/n) - 1
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = (rets.mean() * 252 - 0.04) / vol if vol > 0 else 0
    bench_vol = bench_rets.std() * np.sqrt(252) if len(bench_rets) > 1 else 0
    bench_sharpe = (bench_rets.mean() * 252 - 0.04) / bench_vol if bench_vol > 0 else 0
    eq = (1 + rets).cumprod()
    max_dd = ((eq / eq.cummax()) - 1).min() if len(eq) > 0 else 0

    metrics = {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "days_live": n,
        "live_start": LIVE_START,
        "live_end": LIVE_END,
        "initial_capital": INITIAL_CAPITAL,
        "current_value": float(df["value"].iloc[-1]),
        "benchmark_value": float(df["benchmark"].iloc[-1]),
        "total_return_pct": float((df["value"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "benchmark_return_pct": float((df["benchmark"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "excess_return_pct": float(((df["value"].iloc[-1] - df["benchmark"].iloc[-1]) / INITIAL_CAPITAL) * 100),
        "cagr_pct": float(cagr * 100),
        "benchmark_cagr_pct": float(bench_cagr * 100),
        "sharpe": float(sharpe),
        "benchmark_sharpe": float(bench_sharpe),
        "vol_annual_pct": float(vol * 100),
        "max_drawdown_pct": float(max_dd * 100),
    }
    _save_json(METRICS_PATH, metrics)


if __name__ == "__main__":
    res = update_paper_book()
    print(json.dumps(res, indent=2, default=str))
