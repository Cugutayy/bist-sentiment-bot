"""Günlük top-N shortlist üretimi — "neden" açıklamasıyla.

Akış:
1) En son fiyat verisi + features (lag=False, son güne kadar)
2) En son model artifact'ını yükle (haftalık retrain edilir)
3) Bugün için predict_proba → top-N ticker
4) Her seçim için "neden": en güçlü feature'ları SHAP-benzeri basit
   importance ile açıkla (LightGBM feature_importances)

Çıktı: data/processed/shortlist_YYYY-MM-DD.parquet + console rapor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS
from features.feature_engineering import build_features, feature_columns
from features.loader import load_panel

ARTIFACTS = ROOT / "models" / "artifacts"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


@dataclass
class ShortlistItem:
    ticker: str
    score: float
    rank: int
    last_close: float
    last_return_5d_pct: float
    momentum_60d_pct: float
    rsi_14: float
    rationale: str           # neden seçildi (top feature'lar)


def _find_latest_model() -> Path | None:
    """En son eğitilmiş model artifact'ını bul (son fold)."""
    runs = sorted(ARTIFACTS.glob("*/"), reverse=True)
    for run in runs:
        folds = sorted(run.glob("fold_*.joblib"), reverse=True)
        if folds:
            return folds[0]
    return None


def _explain(estimator, feat_cols: list[str], row: pd.Series, top_k: int = 3) -> str:
    """Basit feature importance + bu ticker için top katkı veren feature'lar."""
    importances = estimator.feature_importances_
    # Importance × feature değeri (büyük ve pozitif = sinyalin neden geldiği)
    contrib = []
    for col, imp in zip(feat_cols, importances):
        val = row.get(col, np.nan)
        if pd.notna(val):
            contrib.append((col, imp, val, imp * abs(val)))
    contrib.sort(key=lambda x: x[3], reverse=True)
    top = contrib[:top_k]
    parts = []
    for col, imp, val, _ in top:
        parts.append(f"{col}={val:+.3f}")
    return " · ".join(parts)


def generate_shortlist(
    as_of: str | None = None,
    top_n: int | None = None,
) -> list[ShortlistItem]:
    """Bugün (veya as_of tarihinde) top-N shortlist üret.

    as_of None ise en son veri tarihini kullanır.
    """
    top_n = top_n or SETTINGS["strategy"]["top_n"]

    model_path = _find_latest_model()
    if model_path is None:
        raise RuntimeError(
            "Eğitilmiş model yok. Önce 'python main.py backtest' (model artifact üretir) "
            "veya 'python main.py retrain' (Faz 3'te eklenecek)."
        )
    estimator = joblib.load(model_path)
    meta = json.loads((model_path.parent / model_path.name.replace(".joblib", "_meta.json")).read_text())
    feat_cols = meta["feature_cols"]
    logger.info(f"Model yüklendi: {model_path.name} (train: {meta['train_period'] if 'train_period' in meta else meta['train_end']})")

    panel = load_panel()
    features = build_features(panel, lag=False)   # canlı predict için lag yok

    # Halisünasyon defansı: sentiment feature'ları NaN kalabilir (data yok),
    # bu satırı atmamalıyız. Sadece fiyat-based feature'lardaki NaN'leri at.
    price_feat_cols = [c for c in feat_cols if not c.startswith("sent_")]
    features = features.dropna(subset=price_feat_cols)

    if as_of is None:
        as_of_dt = features["date"].max()
    else:
        as_of_dt = pd.to_datetime(as_of)
    snapshot = features[features["date"] == as_of_dt].copy()
    if snapshot.empty:
        raise RuntimeError(f"Tarih {as_of_dt.date()} için veri yok")

    X = snapshot[feat_cols]
    proba = estimator.predict_proba(X)[:, 1]
    snapshot["score"] = proba
    ranked = snapshot.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    items: list[ShortlistItem] = []
    for i, row in ranked.iterrows():
        items.append(ShortlistItem(
            ticker=row["ticker"],
            score=float(row["score"]),
            rank=i + 1,
            last_close=float(row["close"]),
            last_return_5d_pct=float(row.get("ret_5", 0)) * 100,
            momentum_60d_pct=float(row.get("momentum_60", 0)) * 100,
            rsi_14=float(row.get("rsi_14", 50)),
            rationale=_explain(estimator, feat_cols, row),
        ))

    # Disk'e yaz
    out_path = PROCESSED / f"shortlist_{as_of_dt.date()}.parquet"
    ranked.to_parquet(out_path, index=False)
    logger.info(f"Shortlist yazıldı → {out_path.name}")
    return items
