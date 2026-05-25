"""LightGBM training — purged walk-forward.

Strateji:
- Target: triple barrier label (-1, 0, +1) — binarize: y = (label == +1)
- Model: LightGBM binary classifier
- Output: predict_proba → cross-sectional ranking sinyali
- Model artifact: models/artifacts/<run_id>/fold_<i>.joblib

Walk-forward: backtest/walk_forward.py'deki splits.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS
from features.feature_engineering import feature_columns
from models.triple_barrier import triple_barrier_labels

ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainedModel:
    estimator: lgb.LGBMClassifier
    feature_cols: list[str]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train_samples: int
    n_features: int


def build_training_set(panel_with_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Feature panelinden (X, y, feature_cols) çıkar.

    Label: ticker bazında triple barrier → binarize (y=1 iff label==+1).
    Tüm NaN satırlar atılır.
    """
    panel = panel_with_features.copy().sort_values(["ticker", "date"])

    # Ticker bazında label hesabı (close üzerinden triple barrier)
    panel["label"] = panel.groupby("ticker", group_keys=False)["close"].apply(
        lambda s: triple_barrier_labels(s)
    )
    panel["target"] = (panel["label"] == 1).astype(int)

    feat_cols = feature_columns(panel)
    keep = ["date", "ticker", "target"] + feat_cols
    panel = panel[keep].dropna()

    X = panel[feat_cols]
    y = panel["target"]
    return panel, y, feat_cols


def train_fold(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
) -> TrainedModel:
    """Tek bir fold için model eğit."""
    cfg_params = (params or SETTINGS["model"]["params"]).copy()
    cfg_params.setdefault("verbose", -1)

    X = train_df[feat_cols]
    y = train_df["target"]

    if y.nunique() < 2:
        raise RuntimeError("Train set tek sınıf — etiket çeşitliliği yok")

    model = lgb.LGBMClassifier(**cfg_params)
    model.fit(X, y)

    return TrainedModel(
        estimator=model,
        feature_cols=feat_cols,
        train_start=train_df["date"].min(),
        train_end=train_df["date"].max(),
        n_train_samples=len(train_df),
        n_features=len(feat_cols),
    )


def predict_fold(tm: TrainedModel, test_df: pd.DataFrame) -> pd.DataFrame:
    """test_df üzerinde predict_proba → 'signal' kolonu (0..1)."""
    X = test_df[tm.feature_cols]
    proba = tm.estimator.predict_proba(X)[:, 1]
    out = test_df[["date", "ticker"]].copy()
    out["signal"] = proba
    return out


def save_model(tm: TrainedModel, run_id: str, fold_idx: int) -> Path:
    """Model artifact'ını diske yaz — reproducibility için sha + meta."""
    folder = ARTIFACTS / run_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"fold_{fold_idx:02d}.joblib"
    joblib.dump(tm.estimator, path)
    meta = {
        "fold_idx": fold_idx,
        "train_start": str(tm.train_start.date()),
        "train_end": str(tm.train_end.date()),
        "n_train_samples": tm.n_train_samples,
        "n_features": tm.n_features,
        "feature_cols": tm.feature_cols,
        "params": SETTINGS["model"]["params"],
    }
    (folder / f"fold_{fold_idx:02d}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def make_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cfg_hash = hashlib.sha256(json.dumps(SETTINGS["model"], sort_keys=True).encode()).hexdigest()[:8]
    return f"{ts}_{cfg_hash}"
