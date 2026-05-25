"""Feature engineering — her feature ayrı fonksiyon, hepsi t-1 lagged.

Look-ahead defansı kritik: her feature günün **kapanışından sonra**
hesaplanır, bir sonraki gün için sinyale dönüştürülür. shift(1)
zorunlu, asla geçmiş + bugün karışmaz.

Çıktı: panel + feature kolonları (long format).
Sentiment kolonları null bırakılır eğer sentiment data yoksa
(faz 2'de fiyat-only baseline, faz 3'te sentiment augment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config import SETTINGS


# ─── Tekil feature fonksiyonları (ticker bazlı, time-series) ────────

def f_return(close: pd.Series, window: int) -> pd.Series:
    """N-günlük yüzde getiri."""
    return close.pct_change(window)


def f_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Günlük getirilerin rolling std (lognormal proxy)."""
    rets = close.pct_change()
    return rets.rolling(window).std()


def f_momentum(close: pd.Series, window: int = 60) -> pd.Series:
    """Uzun-dönem momentum — N-günlük getiri."""
    return close.pct_change(window)


def f_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI — overbought/oversold."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta).clip(lower=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def f_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Hacmin rolling z-score'u — anormal aktivite tespiti."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


def f_relative_strength(close: pd.Series, benchmark_close: pd.Series, window: int = 20) -> pd.Series:
    """Hisse'nin benchmark'a göre relatif performansı, N gün."""
    stock_ret = close.pct_change(window)
    bench_ret = benchmark_close.pct_change(window)
    return stock_ret - bench_ret


# ─── Cross-sectional (her tarih için ticker'lar arası) ──────────────

def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Belirli bir feature'ı her tarih için cross-sectional z-score yap."""
    mean = df.groupby("date")[col].transform("mean")
    std = df.groupby("date")[col].transform("std")
    return (df[col] - mean) / std.replace(0, np.nan)


# ─── Ana pipeline ───────────────────────────────────────────────────

def build_features(panel: pd.DataFrame, lag: bool = True) -> pd.DataFrame:
    """Panel + tüm feature kolonları döndür.

    Parameters
    ----------
    panel : pd.DataFrame
        loader.load_panel() çıktısı (date, ticker, OHLCV, benchmark_close).
    lag : bool
        True ise tüm features shift(1) — t günü için sinyal sadece t-1
        kapanışına kadar olan veriyi görür. Backtest için ZORUNLU true.
        False sadece "şu an için sinyal üret" amacı (live predict).

    Returns
    -------
    pd.DataFrame
        panel + feature kolonları, long format.
    """
    cfg = SETTINGS["features"]
    out = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    # Her ticker için zaman serisi feature'ları
    # pandas 3.0 uyumluluğu: tek ticker'da groups.apply DataFrame döner,
    # bu yüzden her feature için ayrı transform/iterate.
    grouped = out.groupby("ticker", group_keys=False, sort=False)

    for w in cfg["return_windows"]:
        out[f"ret_{w}"] = grouped["close"].transform(lambda s: f_return(s, w))

    out["vol_20"] = grouped["close"].transform(lambda s: f_volatility(s, cfg["vol_window"]))
    out["momentum_60"] = grouped["close"].transform(lambda s: f_momentum(s, cfg["momentum_window"]))
    out["rsi_14"] = grouped["close"].transform(lambda s: f_rsi(s, cfg["rsi_window"]))
    out["volume_z"] = grouped["volume"].transform(lambda s: f_volume_zscore(s))

    # rel_strength iki kolon kullanıyor — transform tek seri lazım,
    # bu yüzden ticker-ticker hesaplayıp birleştir.
    rs_chunks = []
    for ticker, g in grouped:
        rs = f_relative_strength(g["close"], g["benchmark_close"], 20)
        rs.index = g.index
        rs_chunks.append(rs)
    out["rel_strength_20"] = pd.concat(rs_chunks).sort_index()

    # Cross-sectional z-score'lar (rank-equivalent normalize)
    for col in ["ret_5", "ret_20", "momentum_60", "rsi_14", "volume_z", "rel_strength_20"]:
        out[f"cs_{col}"] = cs_zscore(out, col)

    feature_cols = [c for c in out.columns
                    if c.startswith(("ret_", "vol_", "momentum_", "rsi_",
                                     "volume_z", "rel_strength_", "cs_"))]

    if lag:
        # Look-ahead defansı: t günü için sadece t-1 verisi
        out[feature_cols] = grouped[feature_cols].shift(1)

    logger.info(f"Features built: {len(feature_cols)} kolon, {len(out):,} satır")
    return out


def feature_columns(panel_with_features: pd.DataFrame) -> list[str]:
    """Feature kolonlarını otomatik tespit et (model.fit için)."""
    return [c for c in panel_with_features.columns
            if c.startswith(("ret_", "vol_", "momentum_", "rsi_",
                             "volume_z", "rel_strength_", "cs_"))]
