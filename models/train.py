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
from datetime import UTC, datetime
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


def build_training_set(
    panel_with_features: pd.DataFrame,
    horizon_override: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Feature panelinden (X, y, feature_cols) çıkar.

    Label tipi config'den seçilir:
      - 'triple_barrier' (default): TP / SL / time → binarize
      - 'relative_outperform': forward N-day return > cross-sectional median
        → model SECTION'l alpha öğrenir (XU100 buy-and-hold'u GEÇEN hisseler)

    horizon_override: relative_outperform için horizon zorlanır (multi-horizon
    ensemble için kullanılır).
    """
    panel = panel_with_features.copy().sort_values(["ticker", "date"])

    label_type = SETTINGS.get("labels", {}).get("label_type", "triple_barrier")

    if label_type == "relative_outperform":
        horizon = horizon_override or SETTINGS.get("labels", {}).get("relative_horizon_days", 5)
        # Her ticker için N-gün forward return
        panel["fwd_ret"] = panel.groupby("ticker", group_keys=False)["close"].apply(
            lambda s: s.pct_change(horizon).shift(-horizon)
        )
        # Her tarih için cross-section median
        panel["xs_median"] = panel.groupby("date")["fwd_ret"].transform("median")
        # Target: bu ticker median'in üstünde mi?
        panel["target"] = (panel["fwd_ret"] > panel["xs_median"]).astype(int)
        # fwd_ret NaN ise (son N gün) atılmalı
        panel = panel[panel["fwd_ret"].notna()].copy()
    else:
        # Klasik triple barrier
        panel["label"] = panel.groupby("ticker", group_keys=False)["close"].apply(
            lambda s: triple_barrier_labels(s)
        )
        panel["target"] = (panel["label"] == 1).astype(int)

    feat_cols = feature_columns(panel)
    keep = ["date", "ticker", "target"] + feat_cols
    panel = panel[keep]

    # ÖNEMLİ halisünasyon defansı:
    # - CORE feature'lar (ret_5, ret_20, rsi_14, vol_20) NaN ise satırı at.
    #   Bunlar warm-up dönemi ardından her zaman dolu olmalı.
    # - UZUN-horizon feature'lar (mom_250, mom_12_1, high_prox_52w) ilk
    #   yıl NaN olabilir → atmazsak ilk fold'da bu feature'ları kullanmaz,
    #   sonraki fold'larda kullanır. LightGBM NaN'i missing branch öğrenir.
    # - sentiment feature'ları (sent_*) NaN olabilir — model handle eder.
    # - target NaN olamaz.
    core_feat_cols = [c for c in feat_cols if c in ("ret_5", "ret_20", "rsi_14", "vol_20")]
    panel = panel.dropna(subset=["target"] + core_feat_cols)

    X = panel[feat_cols]
    y = panel["target"]
    return panel, y, feat_cols


def train_fold(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
) -> TrainedModel:
    """Tek bir fold için model eğit. Ensemble desteği: birden fazla
    seed ile eğit, predict'te ortalama al. Avramov & Wermers (2006)
    'Investing in Mutual Funds when Returns are Predictable' - model
    uncertainty averaging Sharpe'ı artırır."""
    cfg_params = (params or SETTINGS["model"]["params"]).copy()
    cfg_params.setdefault("verbose", -1)
    n_ensemble = SETTINGS["model"].get("n_ensemble", 1)

    X = train_df[feat_cols]
    y = train_df["target"]

    if y.nunique() < 2:
        raise RuntimeError("Train set tek sınıf — etiket çeşitliliği yok")

    # Recency weighting: son sample'lar daha önemli (exponential decay)
    # Asness et al. 2014: market regime shifts → recent data daha tahmin edici
    use_recency = SETTINGS["model"].get("recency_weighting", False)
    if use_recency and "date" in train_df.columns:
        # 0 (en eski) -> 1 (en yeni) lineer, sonra exp(α * w)
        dates_sorted = train_df["date"].sort_values().unique()
        date_to_w = {d: i / max(1, len(dates_sorted) - 1) for i, d in enumerate(dates_sorted)}
        alpha = SETTINGS["model"].get("recency_alpha", 1.5)  # higher = more emphasis recent
        sample_w = train_df["date"].map(date_to_w).apply(lambda x: np.exp(alpha * x))
        sample_w = sample_w / sample_w.mean()  # normalize
    else:
        sample_w = None

    if n_ensemble > 1:
        # Bagging ensemble - different random_state for diversity
        estimators = []
        for i in range(n_ensemble):
            p = cfg_params.copy()
            p["random_state"] = cfg_params.get("random_state", 42) + i * 1000
            # subsample diversity
            p["subsample"] = 0.85
            p["subsample_freq"] = 1
            p["colsample_bytree"] = 0.85
            m = lgb.LGBMClassifier(**p)
            if sample_w is not None:
                m.fit(X, y, sample_weight=sample_w)
            else:
                m.fit(X, y)
            estimators.append(m)
        # Pack ensemble as a wrapper
        estimator = _EnsembleWrapper(estimators)
    else:
        estimator = lgb.LGBMClassifier(**cfg_params)
        if sample_w is not None:
            estimator.fit(X, y, sample_weight=sample_w)
        else:
            estimator.fit(X, y)

    return TrainedModel(
        estimator=estimator,
        feature_cols=feat_cols,
        train_start=train_df["date"].min(),
        train_end=train_df["date"].max(),
        n_train_samples=len(train_df),
        n_features=len(feat_cols),
    )


class _EnsembleWrapper:
    """LightGBM modeller listesi, predict_proba ortalamasını verir."""
    def __init__(self, estimators):
        self.estimators = estimators
        # ilk modelin attribute'larına proxy
        self.feature_importances_ = estimators[0].feature_importances_
    def predict_proba(self, X):
        import numpy as np
        probs = [m.predict_proba(X) for m in self.estimators]
        return np.mean(probs, axis=0)


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
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    cfg_hash = hashlib.sha256(json.dumps(SETTINGS["model"], sort_keys=True).encode()).hexdigest()[:8]
    return f"{ts}_{cfg_hash}"
