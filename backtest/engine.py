"""Walk-forward backtest engine — gerçek kâr/zarar simülasyonu.

Akış:
1) Panel + features yükle
2) Walk-forward fold'larını üret (purged + embargoed)
3) Her fold için:
   - train_df → LightGBM eğit
   - test_df → predict_proba ile sinyal
4) Sinyallerden günlük portföy oluştur:
   - top-N long, equal weight
   - max_position_pct cap
   - sektör cap (Faz 3'te)
5) Net günlük getiri = pozisyon × ertesi gün return − turnover × cost
6) Tüm fold'lar concat → metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from backtest.costs import CostModel, DEFAULT as DEFAULT_COSTS
from backtest.metrics import summary
from backtest.walk_forward import WalkForwardSplit, make_splits, slice_panel
from config import SETTINGS
from features.feature_engineering import build_features
from features.loader import load_panel
from models.train import build_training_set, predict_fold, train_fold


@dataclass
class BacktestResult:
    daily_returns: pd.Series           # net günlük portföy getirisi
    daily_weights: pd.DataFrame        # date × ticker pozisyon ağırlık matrisi
    fold_metrics: list[dict]           # her fold için ayrı metrik
    overall_metrics: dict              # tüm dönemde toplam
    benchmark_returns: pd.Series       # XU100 buy-and-hold karşılaştırma
    benchmark_metrics: dict


def _build_positions(signals: pd.DataFrame, top_n: int, max_pos: float) -> pd.DataFrame:
    """Sinyallerden günlük long-only equal-weight pozisyon ağırlıkları üret.

    signals: date | ticker | signal (0..1)
    Çıktı: date × ticker pivot, değerler ağırlıklar (sum ≤ 1).
    """
    if signals.empty:
        return pd.DataFrame()
    s = signals.copy()
    # Her gün top-N en yüksek sinyal
    s["rank"] = s.groupby("date")["signal"].rank(ascending=False, method="first")
    s = s[s["rank"] <= top_n]
    n_per_day = s.groupby("date").size()
    s = s.merge(n_per_day.rename("n_picked"), left_on="date", right_index=True)
    s["weight"] = (1.0 / s["n_picked"]).clip(upper=max_pos)
    pivot = s.pivot(index="date", columns="ticker", values="weight").fillna(0)
    return pivot


def _daily_pnl(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    cost: CostModel,
) -> pd.Series:
    """Pozisyon ağırlıklarından net günlük portföy getirisi.

    Mantık:
    - Sinyal t-1 günü kapanışta üretilir (build_features lag=True).
    - Pozisyon t günü açılır, t günü kapanışına kadar tutulur.
    - Net daily return = weights[t] × rets[t] - turnover_cost.

    Sadece weights'in tanımlı olduğu (test) günler için PnL hesaplanır;
    boş günler 0 ile doldurulup Sharpe'ı bozmaz — out-of-sample (out-of-market)
    olan günler hiç katılmaz.
    """
    rets = panel.pivot(index="date", columns="ticker", values="close").pct_change()
    # Weights'in tanımlı olduğu tarih × ticker kesişimi
    common_dates = weights.index.intersection(rets.index)
    common_cols = weights.columns.intersection(rets.columns)
    w = weights.loc[common_dates, common_cols].fillna(0)
    r = rets.loc[common_dates, common_cols].fillna(0)

    gross = (w * r).sum(axis=1)
    # Turnover — ilk gün full entry
    diff = w.diff()
    diff.iloc[0] = w.iloc[0]
    turnover = diff.abs().sum(axis=1)
    cost_pct = turnover * cost.one_way_cost()
    return gross - cost_pct


def _benchmark_returns(panel: pd.DataFrame) -> pd.Series:
    """XU100 günlük getirisi (buy-and-hold karşılaştırma)."""
    bm = panel[["date", "benchmark_close"]].drop_duplicates(subset="date").sort_values("date")
    bm = bm.set_index("date")
    return bm["benchmark_close"].pct_change().fillna(0)


def run_backtest(
    start: str | None = None,
    end: str | None = None,
    top_n: int | None = None,
    cost: CostModel = DEFAULT_COSTS,
) -> BacktestResult:
    """Tam pipeline — panel → features → walk-forward → backtest."""
    top_n = top_n or SETTINGS["strategy"]["top_n"]
    max_pos = SETTINGS["strategy"]["max_position_pct"]

    logger.info("Backtest başlıyor...")
    panel = load_panel(start=start, end=end)
    features = build_features(panel, lag=True)
    train_df_all, _, feat_cols = build_training_set(features)
    logger.info(f"Training set hazır: {len(train_df_all):,} satır × {len(feat_cols)} feature")

    dates = pd.DatetimeIndex(sorted(train_df_all["date"].unique()))
    splits = make_splits(dates)
    if not splits:
        raise RuntimeError("Walk-forward split üretilemedi — veri kısa olabilir")

    all_signals: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    for i, split in enumerate(splits):
        train = train_df_all[(train_df_all["date"] >= split.train_start) &
                             (train_df_all["date"] <= split.train_end)]
        test = train_df_all[(train_df_all["date"] >= split.test_start) &
                            (train_df_all["date"] < split.test_end)]
        if train.empty or test.empty or train["target"].nunique() < 2:
            logger.warning(f"Fold {i}: yetersiz veri ({len(train)} train, {len(test)} test)")
            continue
        try:
            tm = train_fold(train, feat_cols)
        except Exception as e:
            logger.error(f"Fold {i}: training fail → {e}")
            continue
        sig = predict_fold(tm, test)
        all_signals.append(sig)
        fold_metrics.append({
            "fold": i,
            "train_n": len(train),
            "test_n": len(test),
            "train_period": f"{split.train_start.date()} → {split.train_end.date()}",
            "test_period":  f"{split.test_start.date()} → {split.test_end.date()}",
            "positive_rate_train": float(train["target"].mean()),
        })
        logger.info(f"Fold {i}: train={len(train)} test={len(test)} "
                    f"signals={len(sig)} pos_rate={train['target'].mean():.3f}")

    if not all_signals:
        raise RuntimeError("Hiç başarılı fold yok")

    signals_concat = pd.concat(all_signals, ignore_index=True)
    weights = _build_positions(signals_concat, top_n=top_n, max_pos=max_pos)

    # Net günlük getiri (panel: orijinal price panel, NOT features)
    net = _daily_pnl(weights, panel, cost)
    # Benchmark: SADECE backtest test günlerinde, apples-to-apples karşılaştırma
    bench = _benchmark_returns(panel).reindex(net.index).fillna(0)

    overall = summary(net, weights=weights)
    bench_summary = summary(bench)

    logger.info(f"Backtest bitti: {len(splits)} fold, {len(net)} işlem günü")
    return BacktestResult(
        daily_returns=net,
        daily_weights=weights,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        benchmark_returns=bench,
        benchmark_metrics=bench_summary,
    )


def print_report(result: BacktestResult, cost: CostModel = DEFAULT_COSTS) -> None:
    """Konsola backtest raporu."""
    print()
    print("=" * 70)
    print("BACKTEST RAPORU")
    print("=" * 70)
    print(f"Cost modeli: {cost}")
    print(f"Walk-forward fold: {len(result.fold_metrics)}")
    if result.fold_metrics:
        print(f"Dönem: {result.fold_metrics[0]['test_period']} → {result.fold_metrics[-1]['test_period']}")
    print()
    print(f"{'Metrik':<15} {'Portföy':>14} {'XU100 BH':>14}")
    print("-" * 45)
    o = result.overall_metrics
    b = result.benchmark_metrics
    for k in ("sharpe", "cagr", "max_dd", "hit_rate", "mean_d", "std_d"):
        portfo = o.get(k, "—")
        bench = b.get(k, "—")
        print(f"{k:<15} {portfo:>14} {bench:>14}")
    print()
    print(f"n_days:        {o['n_days']:>14} {b['n_days']:>14}")
    if "turnover_d" in o:
        print(f"turnover_d:    {o['turnover_d']:>14}")
    print()
    print("Fold breakdown:")
    for fm in result.fold_metrics:
        print(f"  Fold {fm['fold']:>2}: test={fm['test_period']} "
              f"n_train={fm['train_n']:>5} pos_rate={fm['positive_rate_train']:.3f}")
    print()
