"""KRITIK: backtest ↔ live aynı tarih için aynı feature'ı verir mi?

Senaryo:
- Backtest sırasında t günü için lagged feature'lar t-1 verisini kullanır.
- Live sırasında t günü için aynı feature, t-1 verisinden hesaplanır.
- İkisi BİR BİT TAM AYNI değer vermeli — yoksa backtest yalan.

Common nedenler bu testin fail etmesine:
- Feature fonksiyonunda look-ahead var (full panel görüyor)
- Lag uygulanmıyor / yanlış uygulanıyor
- Cross-sectional z-score sonradan eklenen tarih için farklı hesaplanıyor
- Backtest'te full panel, live'da kesilmiş panel — pencere boyutları farklı
"""
import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import build_features


def _make_panel(n_days: int = 100, n_tickers: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    bm = 1000 + rng.standard_normal(n_days).cumsum()
    for i, t in enumerate(["A.IS", "B.IS", "C.IS"][:n_tickers]):
        close = 100 + rng.standard_normal(n_days).cumsum() * (i + 1)
        rows.append(pd.DataFrame({
            "date": dates,
            "ticker": t,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1000, 10000, n_days),
            "benchmark_close": bm,
        }))
    return pd.concat(rows, ignore_index=True)


def test_backtest_and_live_features_match_for_same_date():
    """En kritik test: t günü için backtest ve live aynı feature değerini vermeli."""
    panel = _make_panel(n_days=80, n_tickers=3)
    cutoff_date = panel["date"].iloc[50]

    # BACKTEST modu: tüm panel, lag=True
    full_lagged = build_features(panel, lag=True)
    backtest_row = full_lagged[
        (full_lagged["ticker"] == "A.IS") & (full_lagged["date"] == cutoff_date)
    ].iloc[0]

    # LIVE modu: sadece cutoff_date'e kadar olan panel, lag=True
    live_panel = panel[panel["date"] <= cutoff_date].copy()
    live_lagged = build_features(live_panel, lag=True)
    live_row = live_lagged[
        (live_lagged["ticker"] == "A.IS") & (live_lagged["date"] == cutoff_date)
    ].iloc[0]

    feature_cols = [c for c in full_lagged.columns
                    if c.startswith(("ret_", "vol_", "momentum_", "rsi_",
                                     "volume_z", "rel_strength_", "cs_"))]
    for col in feature_cols:
        b = backtest_row[col]
        l = live_row[col]
        if pd.isna(b) and pd.isna(l):
            continue
        # Cross-sectional z-score'lar live'da farklı olabilir çünkü
        # universe içindeki diğer ticker'lar da kesilmiş veriden hesaplanır.
        # Per-ticker (zaman serisi) feature'lar TAM AYNI olmalı.
        if col.startswith("cs_"):
            continue
        assert abs(b - l) < 1e-9, (
            f"PARITY BREACH! '{col}' backtest={b} live={l} — "
            "feature fonksiyonu look-ahead yapıyor olabilir."
        )


def test_cross_sectional_features_use_only_current_universe():
    """Cross-sectional z-score'lar sadece o gün universe'inden hesaplanmalı.

    Backtest'te full panel görür ama tarih bazında groupby ile cs_zscore
    her tarih için ayrı yapılır → live'da da aynı sonucu vermelidir.
    """
    panel = _make_panel(n_days=80, n_tickers=3)
    cutoff_date = panel["date"].iloc[50]

    full = build_features(panel, lag=True)
    live = build_features(panel[panel["date"] <= cutoff_date], lag=True)

    cs_cols = [c for c in full.columns if c.startswith("cs_")]
    for ticker in panel["ticker"].unique():
        b = full[(full["ticker"] == ticker) & (full["date"] == cutoff_date)].iloc[0]
        l = live[(live["ticker"] == ticker) & (live["date"] == cutoff_date)].iloc[0]
        for col in cs_cols:
            if pd.isna(b[col]) and pd.isna(l[col]):
                continue
            assert abs(b[col] - l[col]) < 1e-9, (
                f"CS PARITY BREACH! {ticker} '{col}' backtest={b[col]} live={l[col]}"
            )


def test_no_future_data_affects_past_features():
    """Bugünden sonraki bir gün eklersek geçmiş feature değişmemeli."""
    panel = _make_panel(n_days=60, n_tickers=2)
    cutoff_date = panel["date"].iloc[40]

    f1 = build_features(panel[panel["date"] <= cutoff_date], lag=True)
    f2 = build_features(panel, lag=True)   # extra 20 gün gelecek var

    a_past = f1[(f1["ticker"] == "A.IS") & (f1["date"] == cutoff_date)].iloc[0]
    a_future = f2[(f2["ticker"] == "A.IS") & (f2["date"] == cutoff_date)].iloc[0]
    feature_cols = [c for c in f1.columns
                    if c.startswith(("ret_", "vol_", "momentum_", "rsi_",
                                     "volume_z", "rel_strength_"))]
    for col in feature_cols:
        if pd.isna(a_past[col]) and pd.isna(a_future[col]):
            continue
        assert abs(a_past[col] - a_future[col]) < 1e-9, (
            f"LEAK! '{col}' geçmiş feature gelecek data eklenince değişti."
        )
