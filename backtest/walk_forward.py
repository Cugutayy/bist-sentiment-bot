"""Walk-forward backtest engine — purged + embargoed.

Standart sklearn TimeSeriesSplit data leak'e karşı yeterli DEĞİL çünkü
overlapping labels (örn. 5-gün horizon) train→test geçişinde sızıntı
yaratır. López de Prado'nun önerdiği iki katman:

1) **Purge**: train setinin sonunda, test ile zaman bakımından örtüşen
   örnekleri at.
2) **Embargo**: test biter bitmez train'e tekrar veri eklemeden önce
   N gün boşluk koy (autocorrelation savunması).

İskelet — Faz 2'de feature pipeline ve model bağlanacak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from loguru import logger

from backtest.costs import CostModel, DEFAULT as DEFAULT_COSTS
from config import SETTINGS


@dataclass
class WalkForwardSplit:
    """Tek bir walk-forward iterasyonu için tarih aralıkları."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_splits(
    dates: pd.DatetimeIndex,
    train_window_days: int | None = None,
    test_window_days: int | None = None,
    purge_days: int | None = None,
    embargo_days: int | None = None,
) -> list[WalkForwardSplit]:
    """Tarih ekseninde rolling pencere üret."""
    wf = SETTINGS["model"]["walk_forward"]
    train_window_days = train_window_days or wf["train_window_days"]
    test_window_days  = test_window_days  or wf["test_window_days"]
    purge_days        = purge_days        or wf["purge_days"]
    embargo_days      = embargo_days      or wf["embargo_days"]

    dates = pd.DatetimeIndex(sorted(set(dates)))
    if len(dates) == 0:
        return []

    splits: list[WalkForwardSplit] = []
    start = dates[0]
    while True:
        train_start = start
        train_end_target = train_start + pd.Timedelta(days=train_window_days)
        # Purge: test ile çakışacak son N gün train'den at
        train_end = train_end_target - pd.Timedelta(days=purge_days)
        test_start = train_end_target
        test_end = test_start + pd.Timedelta(days=test_window_days)

        if test_end > dates[-1]:
            break

        splits.append(WalkForwardSplit(train_start, train_end, test_start, test_end))
        # Embargo: bir sonraki iterasyon test_end + embargo'dan başla
        start = test_end + pd.Timedelta(days=embargo_days)

    logger.info(f"Walk-forward: {len(splits)} fold üretildi "
                f"(train={train_window_days}d, test={test_window_days}d, "
                f"purge={purge_days}d, embargo={embargo_days}d)")
    return splits


def slice_panel(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """date kolonuna göre [start, end) aralığı."""
    return df[(df["date"] >= start) & (df["date"] < end)].copy()


@dataclass
class FoldResult:
    fold_idx: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    daily_returns: pd.Series   # net (cost sonrası) günlük return
    n_train: int
    n_test: int


def run_walk_forward(
    panel: pd.DataFrame,                # date, ticker, return + feature kolonları
    train_fn: Callable,                  # (train_df) -> model
    predict_fn: Callable,                # (model, test_df) -> sinyaller [-1, 0, +1] veya 0..1 prob
    strategy_fn: Callable,               # (sinyaller, panel) -> günlük position weights
    cost_model: CostModel = DEFAULT_COSTS,
) -> dict:
    """Üst seviye walk-forward orchestrator.

    Faz 2'de gerçek feature engineering + LightGBM model bağlanacak.
    Şimdilik mimari kontur — testlerden geçecek minimum implementasyon
    bir sonraki commit'te.
    """
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    splits = make_splits(dates)

    results: list[FoldResult] = []
    for i, split in enumerate(splits):
        train_df = slice_panel(panel, split.train_start, split.train_end)
        test_df = slice_panel(panel, split.test_start, split.test_end)

        if train_df.empty or test_df.empty:
            logger.warning(f"Fold {i}: empty data, atlanıyor")
            continue

        model = train_fn(train_df)
        sig = predict_fn(model, test_df)
        weights = strategy_fn(sig, test_df)

        # Net günlük getiri = pozisyon × ertesi gün return − turnover × cost
        # (Faz 2'de detaylı hesap — şimdilik placeholder)
        daily = pd.Series(dtype=float)

        results.append(FoldResult(
            fold_idx=i,
            test_start=split.test_start,
            test_end=split.test_end,
            daily_returns=daily,
            n_train=len(train_df),
            n_test=len(test_df),
        ))
        logger.info(f"Fold {i}: train={len(train_df)} test={len(test_df)}")

    # Tüm fold'ları concat et + raporla
    all_returns = pd.concat([r.daily_returns for r in results]) if results else pd.Series(dtype=float)
    metrics = compute_metrics(all_returns) if not all_returns.empty else {}
    metrics["n_folds"] = len(results)
    metrics["cost_model"] = str(cost_model)
    return {"fold_results": results, "metrics": metrics}


def compute_metrics(returns: pd.Series) -> dict:
    """Sharpe, max DD, hit rate, CAGR."""
    if returns.empty or returns.std() == 0:
        return {"sharpe": float("nan"), "max_dd": float("nan"), "cagr": float("nan")}
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    eq = (1 + returns).cumprod()
    dd = ((eq / eq.cummax()) - 1).min()
    years = len(returns) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    hit = (returns > 0).mean()
    return {
        "sharpe": round(sharpe, 3),
        "max_dd": round(dd, 3),
        "cagr": round(cagr, 3),
        "hit_rate": round(hit, 3),
        "n_obs": len(returns),
    }


if __name__ == "__main__":
    # Hızlı sanity — gerçek panel olmadan sadece split üretimini doğrula
    fake_dates = pd.date_range("2021-01-01", "2026-05-01", freq="B")
    splits = make_splits(fake_dates)
    for s in splits[:3]:
        print(s)
    print(f"... toplam {len(splits)} fold")
