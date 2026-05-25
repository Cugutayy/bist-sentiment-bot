"""Halisünasyon defansı testleri.

Kullanıcı uyarısı: "kodun testin botun halisünasyon yaşamasın"
Bu testler her commit'te modelin uydurma sinyallerden korunduğunu doğrular.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import build_features, _add_sentiment_features


def _make_panel(n_days: int = 100, n_tickers: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    bm = 1000 + rng.standard_normal(n_days).cumsum()
    for i, t in enumerate([f"T{i}.IS" for i in range(n_tickers)]):
        close = 100 + rng.standard_normal(n_days).cumsum() * (i + 1)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": close - 0.5, "high": close + 1, "low": close - 1,
            "close": close, "volume": rng.integers(1000, 10000, n_days),
            "benchmark_close": bm,
        }))
    return pd.concat(rows, ignore_index=True)


def test_sentiment_missing_is_nan_not_zero(monkeypatch, tmp_path):
    """Sentiment data yokken sent_* feature'lar NaN olmalı, 0 değil.

    SEBEP: 0 = "neutral sentiment" diye yorumlanır, oysa "data yok"
    farklı bir bilgi. NaN ise LightGBM missing branch öğrenir.
    """
    # SENTIMENT_DAILY mevcut değil — temp dir
    from features import feature_engineering as fe
    monkeypatch.setattr(fe, "SENTIMENT_DAILY", tmp_path / "yok.parquet")

    panel = _make_panel(n_days=50, n_tickers=2)
    out = build_features(panel, lag=False)

    sent_cols = [c for c in out.columns if c.startswith("sent_w")
                 or c.startswith("sent_surprise") or c.startswith("sent_momentum")]
    assert len(sent_cols) > 0, "Sentiment feature kolonları eksik"

    for c in sent_cols:
        assert out[c].isna().all(), \
            f"{c}: sentiment data yokken 0 dolduruluyor — halisünasyon! NaN olmalı."

    # has_data flag 0 olmalı
    assert (out["sent_has_data"] == 0).all(), \
        "sent_has_data flag yanlış: data yok ama 1 gösteriyor"


def test_sentiment_partial_data_no_implicit_zero(monkeypatch, tmp_path):
    """Bazı tarihlerde sentiment varsa, eşleşmeyen satırlar NaN kalmalı (0 değil)."""
    from features import feature_engineering as fe
    sent_path = tmp_path / "sent.parquet"

    # Sadece 1 ticker × 5 tarih için sentiment
    fake_sent = pd.DataFrame({
        "date": pd.date_range("2024-01-15", periods=5, freq="B"),
        "ticker": ["T0.IS"] * 5,
        "sentiment_w": [0.5, 0.3, -0.2, 0.1, 0.0],
        "news_count": [2, 1, 1, 3, 1],
        "sentiment_std": [0.1, 0.0, 0.0, 0.2, 0.0],
        "source_diversity": [2, 1, 1, 2, 1],
    })
    fake_sent.to_parquet(sent_path, index=False)
    monkeypatch.setattr(fe, "SENTIMENT_DAILY", sent_path)

    panel = _make_panel(n_days=50, n_tickers=2)
    out = build_features(panel, lag=False)

    # T1.IS'in hiçbir sentiment'i olmamalı
    t1 = out[out["ticker"] == "T1.IS"]
    assert t1["sent_w_3d"].isna().all(), "T1.IS sentiment data yok ama 0/değer var"
    assert (t1["sent_has_data"] == 0).all()

    # T0.IS'in sentiment tarihlerinde değer olmalı, diğerlerinde NaN
    t0 = out[out["ticker"] == "T0.IS"]
    has_sent = t0[t0["sent_has_data"] == 1]
    no_sent = t0[t0["sent_has_data"] == 0]
    assert len(has_sent) > 0, "Hiçbir tarihte sentiment eşleşmedi"
    if not no_sent.empty:
        assert no_sent["sent_w_3d"].isna().all(), \
            "Sentiment olmayan tarihlerde 0 doldurulmuş — halisünasyon!"


def test_no_feature_columns_have_inf():
    """inf değerler model'i bozar; bölme sıfırla yapılan yerlerde NaN'a clip ediyor muyuz?"""
    panel = _make_panel(n_days=100, n_tickers=3)
    out = build_features(panel, lag=True)
    feature_cols = [c for c in out.columns
                    if c.startswith(("ret_", "vol_", "momentum_", "rsi_",
                                     "volume_z", "rel_strength_", "cs_"))]
    inf_count = ((out[feature_cols] == np.inf) | (out[feature_cols] == -np.inf)).sum().sum()
    assert inf_count == 0, f"{inf_count} inf değer var — divide by zero koruması eksik"


def test_rsi_bounded_0_100():
    """RSI matematiksel olarak [0, 100] aralığında olmalı."""
    from features.feature_engineering import f_rsi
    rng = np.random.default_rng(42)
    s = pd.Series(100 + rng.standard_normal(100).cumsum())
    rsi = f_rsi(s, 14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), \
        f"RSI range bozuk: min={rsi.min()} max={rsi.max()}"


def test_volume_zero_doesnt_break_zscore():
    """Bazı günler volume = 0 olabilir; rolling std de 0 olabilir → div by zero koruma."""
    from features.feature_engineering import f_volume_zscore
    vol = pd.Series([0] * 25)  # sürekli 0
    z = f_volume_zscore(vol, 20)
    # NaN olmalı (std=0), inf değil
    assert not np.isinf(z.dropna()).any(), "Volume z-score inf üretti"


def test_dropna_in_training_doesnt_kill_all_rows_when_sentiment_missing():
    """Sentiment yokken (sent_*=NaN) build_training_set TÜM satırları atmamalı.

    SEBEP: dropna() tüm kolonlara bakarsa, sent_* NaN olduğu için her satır atılır.
    Sadece fiyat-based feature'larda NaN olan satırları atmalı.
    """
    # Bu test'i basit tutmak için import ve fake panel
    from models.train import build_training_set
    from features.feature_engineering import build_features

    panel = _make_panel(n_days=200, n_tickers=3)  # 200 gün ki triple barrier label çıksın
    feats = build_features(panel, lag=True)
    # Sentiment hepsi NaN olmalı (SENTIMENT_DAILY yok)
    sent_cols = [c for c in feats.columns if c.startswith("sent_w")]
    assert feats[sent_cols].isna().all().all(), "Sentiment NaN olmalıydı (sentiment_daily yok)"

    train_df, y, fc = build_training_set(feats)
    # Tüm satırlar atılmamalı (price feature'lar OK olduğunda)
    assert len(train_df) > 0, "Sentiment NaN nedeniyle tüm training set atıldı — halisünasyon defansı bozuk!"
    assert len(train_df) > len(feats) * 0.5, \
        f"Training set %50'den az kaldı: {len(train_df)}/{len(feats)}"
