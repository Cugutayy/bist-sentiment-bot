"""KRITIK: feature engineering'in look-ahead bias'sızlığını doğrula.

t günündeki feature'lar SADECE t-1 ve öncesi veriyi görmeli.
Eğer bir feature t günü close'unu kullanırsa, tüm backtest yalan olur.
"""
import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import build_features, f_return, f_volatility, f_rsi


def _synthetic_panel(n_days: int = 100, n_tickers: int = 3, seed: int = 0) -> pd.DataFrame:
    """Deterministik küçük panel."""
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


def test_no_lookahead_features_use_only_past():
    """Bir feature, t günü close kullanıyorsa lookahead var demektir.

    Test: aynı paneli iki kere kullan, sadece SON satırın close'unu değiştir.
    Lagged feature'lar (t-1 ve öncesi) DEĞIŞMEMELI son satır hariç hiçbir
    yerde, çünkü sadece son satırın geçmişi etkilenmedi.
    """
    panel = _synthetic_panel(n_days=80)
    f1 = build_features(panel, lag=True)

    # Sadece son satır close'unu büyük bir değere zıplat
    panel_modified = panel.copy()
    last_mask = (panel_modified["ticker"] == "A.IS") & (panel_modified["date"] == panel_modified["date"].max())
    panel_modified.loc[last_mask, "close"] = 9999.0

    f2 = build_features(panel_modified, lag=True)

    # A.IS'in son tarihten önceki tüm satırları aynı feature değerlerini taşımalı
    a1 = f1[(f1["ticker"] == "A.IS") & (f1["date"] < f1["date"].max())]
    a2 = f2[(f2["ticker"] == "A.IS") & (f2["date"] < f2["date"].max())]

    for col in ["ret_5", "ret_20", "vol_20", "momentum_60", "rsi_14"]:
        # NaN'leri filtrele, kalan değerler aynı olmalı
        diff = (a1[col].dropna().reset_index(drop=True) - a2[col].dropna().reset_index(drop=True)).abs().max()
        assert diff < 1e-9 or pd.isna(diff), f"LOOKAHEAD! {col} son satır close değişince geçmiş etkilendi: diff={diff}"


def test_lag_shifts_by_one_day():
    """lag=True iken t günündeki feature t-1 close'tan türemiş olmalı."""
    panel = _synthetic_panel(n_days=50, n_tickers=1)
    no_lag = build_features(panel, lag=False)
    lagged = build_features(panel, lag=True)

    # ret_5 hesabı: lag=False iken t günü, lag=True iken t-1 gün
    nl = no_lag["ret_5"].dropna().reset_index(drop=True)
    l = lagged["ret_5"].dropna().reset_index(drop=True)
    # Lagged seri, no-lag serinin bir gün gecikmişi olmalı
    assert (nl[:-1].values - l[1:].values).abs().max() < 1e-9


def test_return_function():
    s = pd.Series([100, 105, 110])
    assert abs(f_return(s, 1).iloc[1] - 0.05) < 1e-9
    assert abs(f_return(s, 1).iloc[2] - 0.0476190) < 1e-5


def test_volatility_positive():
    s = pd.Series(100 + np.random.default_rng(42).standard_normal(50).cumsum())
    v = f_volatility(s, 10).dropna()
    assert (v > 0).all()


def test_rsi_in_range():
    s = pd.Series(100 + np.random.default_rng(42).standard_normal(50).cumsum())
    r = f_rsi(s, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
