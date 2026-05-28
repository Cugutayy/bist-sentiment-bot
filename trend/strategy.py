"""Multi-asset trend-following — production modül.

Tasarım (scripts/test_trend_v2.py'de validate edildi):
  - Universe: 29 likit ETF + kripto (hisse/tahvil/emtia/FX/REIT/crypto)
  - Sinyal: 3 trend göstergesinin blend'i
      1. MA crossover (50 vs 200)
      2. Donchian breakout (50-gün range pozisyonu)
      3. Time-series momentum (60/120/250-gün getiri işareti)
  - Pozisyon: risk-parity (1/vol ağırlık) + portföy-seviyesi vol-target
  - Long-short, günlük sinyal, haftalık-ish rebalance doğal

Look-ahead defansı: tüm sinyaller .shift(1) (t-1 kapanış → t pozisyon).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── Universe ────────────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    # Hisse (bölgesel + sektör)
    "SPY", "QQQ", "IWM", "EFA", "EEM", "XLE", "XLF", "XLK", "XLV", "XLI",
    # Tahvil
    "TLT", "IEF", "SHY", "LQD", "HYG",
    # Emtia
    "GLD", "SLV", "USO", "UNG", "DBC", "DBA", "CPER",
    # FX / Dolar
    "UUP", "FXE", "FXY",
    # REIT + kripto
    "VNQ", "BTC-USD", "ETH-USD",
]

ANN = 252


@dataclass
class TrendConfig:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    ma_fast: int = 50
    ma_slow: int = 200
    breakout_lookback: int = 50
    mom_horizons: tuple[int, ...] = (60, 120, 250)
    vol_lookback: int = 50
    vol_target_annual: float = 0.15
    leverage_cap: float = 5.0
    cost_per_trade: float = 0.0005   # 5bp ETF
    rf_annual: float = 0.03          # USD risk-free


# ── Sinyal göstergeleri ─────────────────────────────────────────────

def signal_ma_cross(close: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """MA crossover: fast > slow → +1, aksi −1."""
    return np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())


def signal_breakout(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Donchian breakout: range içi pozisyon > 0.5 → +1, aksi −1."""
    hi = close.rolling(lookback).max()
    lo = close.rolling(lookback).min()
    pos_in_range = (close - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame(
        np.where(pos_in_range > 0.5, 1.0, -1.0),
        index=close.index, columns=close.columns,
    )


def signal_ts_momentum(close: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Time-series momentum: N-gün getiri işaretlerinin ortalaması."""
    s = sum(np.sign(close.pct_change(h)) for h in horizons) / len(horizons)
    return s


def blended_signal(close: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """3 sinyalin ortalaması, t-1 lagged (look-ahead defansı)."""
    s_ma = signal_ma_cross(close, cfg.ma_fast, cfg.ma_slow)
    s_bo = signal_breakout(close, cfg.breakout_lookback)
    s_mo = signal_ts_momentum(close, cfg.mom_horizons)
    blend = (s_ma + s_bo + s_mo) / 3.0      # -1 .. +1
    return blend.shift(1)


# ── Backtest ────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    daily_returns: pd.Series
    weights: pd.DataFrame
    gross_exposure: pd.Series
    metrics: dict


def run_trend_backtest(close: pd.DataFrame, cfg: TrendConfig | None = None) -> TrendResult:
    """close: date × ticker fiyat paneli (adjusted). → TrendResult.

    Risk-parity + portföy-seviyesi vol-target.
    """
    cfg = cfg or TrendConfig()
    close = close.sort_index()
    rets = close.pct_change()

    sig = blended_signal(close, cfg)

    # Risk-parity: her asset 1/vol ağırlık (eşit risk katkısı)
    cvol = rets.rolling(cfg.vol_lookback).std() * np.sqrt(ANN)
    inv_vol = (1.0 / cvol).clip(0, 50).shift(1)
    pos = sig * inv_vol
    # Normalize: gross exposure = 1
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    pos = pos.div(gross, axis=0).fillna(0.0)

    raw = (pos * rets).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1)
    raw_net = (raw - turn * cfg.cost_per_trade).dropna()

    # Portföy-seviyesi vol-target (ex-post, lagged)
    realized = raw_net.rolling(60, min_periods=20).std() * np.sqrt(ANN)
    scale = (cfg.vol_target_annual / realized).shift(1).clip(0, cfg.leverage_cap).fillna(1.0)
    net = (raw_net * scale).dropna()

    # Ölçeklenmiş ağırlıklar (raporlama)
    weights = pos.reindex(net.index).multiply(scale.reindex(net.index), axis=0).fillna(0.0)
    gross_exp = weights.abs().sum(axis=1)

    return TrendResult(
        daily_returns=net,
        weights=weights,
        gross_exposure=gross_exp,
        metrics=compute_metrics(net, cfg.rf_annual),
    )


def compute_metrics(r: pd.Series, rf_annual: float = 0.03) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"status": "insufficient_data", "n_days": len(r)}
    rf_d = rf_annual / ANN
    cagr = (1 + r).prod() ** (ANN / len(r)) - 1
    sharpe = (r.mean() - rf_d) / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    sortino_dn = r[r < 0].std()
    sortino = (r.mean() - rf_d) / sortino_dn * np.sqrt(ANN) if sortino_dn > 0 else 0.0
    eq = (1 + r).cumprod()
    max_dd = ((eq / eq.cummax()) - 1).min()
    monthly = (1 + r).groupby(pd.Grouper(freq="ME")).prod() - 1
    yearly = (1 + r).groupby(pd.Grouper(freq="YE")).prod() - 1
    return {
        "n_days": int(len(r)),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd": round(float(max_dd), 4),
        "vol_annual": round(float(r.std() * np.sqrt(ANN)), 4),
        "positive_months_pct": round(float((monthly > 0).mean()), 3),
        "positive_years_pct": round(float((yearly > 0).mean()), 3),
        "best_month": round(float(monthly.max()), 4),
        "worst_month": round(float(monthly.min()), 4),
    }
