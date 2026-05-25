"""Backtest weights matrix → trade events + neden açıklaması.

Backtest sırasında her gün hangi ticker alındı/satıldı,
o günkü feature'lar neydi, hangi model'in kararıydı.

Çıktı: dashboard'da görüntülenebilir trade ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from config import ROOT


def extract_trades(weights: pd.DataFrame, weight_threshold: float = 0.001) -> pd.DataFrame:
    """Weights matrix'ten trade event listesi çıkar.

    BUY: weight artışı (yeni pozisyon veya artırma)
    SELL: weight düşüşü (kısmi/tam çıkış)
    """
    if weights.empty:
        return pd.DataFrame()
    w_prev = weights.shift(1).fillna(0)
    diff = weights - w_prev

    trades = []
    for date in diff.index:
        for ticker in diff.columns:
            delta = float(diff.loc[date, ticker])
            if abs(delta) < weight_threshold:
                continue
            trades.append({
                "date": date,
                "ticker": ticker,
                "action": "BUY" if delta > 0 else "SELL",
                "weight_delta": delta,
                "new_weight": float(weights.loc[date, ticker]),
                "prev_weight": float(w_prev.loc[date, ticker]),
            })
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values(["date", "action"]).reset_index(drop=True)
    return df


def explain_trade(
    estimator,
    feat_cols: list[str],
    feature_row: pd.Series,
    top_k: int = 5,
) -> dict:
    """Tek bir trade için 'neden' açıklaması — feature importance × value.

    LightGBM'in importance'sı her feature'ın model'e katkısı.
    İmportance × |value| → o ticker için "en belirleyici" feature'lar.
    """
    importances = estimator.feature_importances_
    contribs = []
    for col, imp in zip(feat_cols, importances):
        val = feature_row.get(col, np.nan)
        if pd.notna(val) and imp > 0:
            contribs.append({
                "feature": col,
                "value": float(val),
                "importance": int(imp),
                "score": int(imp) * abs(float(val)),
            })
    contribs.sort(key=lambda x: x["score"], reverse=True)
    top = contribs[:top_k]

    rationale = " · ".join([f"{c['feature']}={c['value']:+.3f}" for c in top[:3]])
    return {
        "top_features": top,
        "rationale": rationale,
    }


def build_trade_ledger(run_id: str | None = None) -> pd.DataFrame:
    """Backtest run için tam trade ledger + reasoning.

    Reasoning için ARTIFACTS/<run_id>/fold_*.joblib model'ler kullanılır.
    Her trade, o günkü en yakın fold'un model'iyle açıklanır.
    """
    from features.feature_engineering import build_features, feature_columns
    from features.loader import load_panel

    bt_dir = ROOT / "data" / "processed" / "backtests"
    runs = sorted(bt_dir.glob("*/"), reverse=True)
    if not runs:
        return pd.DataFrame()
    if run_id:
        run_dir = bt_dir / run_id
    else:
        run_dir = runs[0]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    weights = pd.read_parquet(run_dir / "weights.parquet").set_index("date")
    weights.index = pd.to_datetime(weights.index)

    trades = extract_trades(weights)
    if trades.empty:
        return trades

    # Fold model'lerini yükle
    models_dir = ROOT / "models" / "artifacts"
    art_runs = sorted(models_dir.glob("*/"), reverse=True)
    if not art_runs:
        logger.warning("Model artifact yok — explain atlanıyor")
        trades["rationale"] = ""
        trades["score"] = float("nan")
        return trades

    art_dir = art_runs[0]
    folds = {}
    for fold_file in art_dir.glob("fold_*.joblib"):
        idx = int(fold_file.stem.split("_")[1])
        meta_file = art_dir / f"{fold_file.stem}_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            est = joblib.load(fold_file)
            folds[idx] = {
                "estimator": est,
                "feat_cols": meta["feature_cols"],
                "test_start": pd.to_datetime(meta.get("test_start", meta.get("train_end"))),
            }

    if not folds:
        trades["rationale"] = ""
        trades["score"] = float("nan")
        return trades

    # Feature panel
    panel = load_panel()
    features = build_features(panel, lag=True)
    feat_cols = feature_columns(features)

    # Her trade için: en yakın fold model'i + o gün feature'ı + reasoning
    rationales = []
    scores = []
    for _, row in trades.iterrows():
        trade_date = row["date"]
        # En yakın fold (genelde aynı fold)
        nearest_fold = min(folds.keys(),
                          key=lambda i: abs((folds[i]["test_start"] - trade_date).days))
        fold = folds[nearest_fold]
        # O gün ticker'a ait feature satırı
        snapshot = features[(features["date"] == trade_date) & (features["ticker"] == row["ticker"])]
        if snapshot.empty:
            rationales.append("")
            scores.append(float("nan"))
            continue
        feature_row = snapshot.iloc[0]
        try:
            X = feature_row[fold["feat_cols"]].to_frame().T
            proba = float(fold["estimator"].predict_proba(X)[0, 1])
        except Exception:
            proba = float("nan")
        scores.append(proba)
        exp = explain_trade(fold["estimator"], fold["feat_cols"], feature_row)
        rationales.append(exp["rationale"])

    trades["score"] = scores
    trades["rationale"] = rationales
    return trades


if __name__ == "__main__":
    df = build_trade_ledger()
    print(df.head(20))
    print(f"\nToplam trade: {len(df)}")
