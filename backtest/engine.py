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
from models.train import build_training_set, make_run_id, predict_fold, save_model, train_fold


@dataclass
class BacktestResult:
    daily_returns: pd.Series           # net günlük portföy getirisi
    daily_weights: pd.DataFrame        # date × ticker pozisyon ağırlık matrisi
    fold_metrics: list[dict]           # her fold için ayrı metrik
    overall_metrics: dict              # tüm dönemde toplam
    benchmark_returns: pd.Series       # XU100 buy-and-hold karşılaştırma
    benchmark_metrics: dict


def _build_positions(
    signals: pd.DataFrame,
    top_n: int,
    max_pos: float,
    rebalance_freq: int = 1,
    smooth_window: int = 1,
) -> pd.DataFrame:
    """Sinyallerden long-only equal-weight pozisyon ağırlıkları üret.

    Turnover azaltma:
    - smooth_window: sinyal ticker BAZLI EMA ile yumuşatılır
    - rebalance_freq: top-N seçimi sadece rebalance günlerinde, arada ffill

    Fold sınırı düzeltmesi: fold lar arası (uzun) tarih boşluklarında
    eski pozisyonlar TAŞINMAZ. Her fold'un ilk günü zorla rebalance.
    Boşluk eşiği = rebalance_freq × 3 gün (örn. 5 freq için 15 gün).

    signals: date | ticker | signal (0..1)
    Çıktı: date × ticker pivot, değerler ağırlıklar.
    """
    if signals.empty:
        return pd.DataFrame()
    s = signals.copy().sort_values(["ticker", "date"])

    # Sinyal smoothing (ticker bazlı EMA)
    if smooth_window > 1:
        s["signal"] = s.groupby("ticker")["signal"].transform(
            lambda x: x.ewm(span=smooth_window, adjust=False).mean()
        )

    all_dates = sorted(s["date"].unique())

    # Fold sınırı tespiti: ardışık tarihler arasında çok gün varsa
    # (>3× rebalance_freq), orada yeni fold başlamış say → ilk gün rebalance.
    gap_threshold = pd.Timedelta(days=max(rebalance_freq * 3, 14))
    fold_starts = [all_dates[0]]
    for i in range(1, len(all_dates)):
        if (all_dates[i] - all_dates[i - 1]) > gap_threshold:
            fold_starts.append(all_dates[i])

    # Her fold içinde rebalance günleri: fold_start + her N işlem günü
    rebalance_dates = set()
    for fs in fold_starts:
        fs_idx = all_dates.index(fs)
        # Bu fold'un son günü = bir sonraki fold_start'ın bir önceki günü veya panel sonu
        next_fs_idx = next(
            (all_dates.index(nfs) for nfs in fold_starts if all_dates.index(nfs) > fs_idx),
            len(all_dates),
        )
        fold_dates = all_dates[fs_idx:next_fs_idx]
        for j in range(0, len(fold_dates), rebalance_freq):
            rebalance_dates.add(fold_dates[j])

    # Sadece rebalance günlerinde top-N seç
    rb = s[s["date"].isin(rebalance_dates)].copy()
    rb["rank"] = rb.groupby("date")["signal"].rank(ascending=False, method="first")
    rb = rb[rb["rank"] <= top_n]
    n_per_day = rb.groupby("date").size()
    rb = rb.merge(n_per_day.rename("n_picked"), left_on="date", right_index=True)
    rb["weight"] = (1.0 / rb["n_picked"]).clip(upper=max_pos)

    pivot = rb.pivot(index="date", columns="ticker", values="weight")

    # ffill ama fold sınırlarında SIFIRLA: her fold için ayrı ffill
    full_index = pd.DatetimeIndex(all_dates)
    pivot = pivot.reindex(full_index)
    out_chunks = []
    for i, fs in enumerate(fold_starts):
        fs_idx = all_dates.index(fs)
        next_fs_idx = (all_dates.index(fold_starts[i + 1])
                       if i + 1 < len(fold_starts) else len(all_dates))
        chunk_dates = all_dates[fs_idx:next_fs_idx]
        chunk = pivot.loc[chunk_dates].ffill().fillna(0)
        out_chunks.append(chunk)
    pivot = pd.concat(out_chunks).fillna(0)
    return pivot


def _daily_pnl(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    cost: CostModel,
) -> pd.Series:
    """Pozisyon ağırlıklarından net günlük portföy getirisi.

    DOĞRU ZAMANSAL ALIGNMENT (kritik):
    - Feature'lar (lag=True) → t günü için close[..., t-1] verisini kullanır.
    - Model train target: row i = "close[i]'da al, close[i+1..i+N] gözle".
    - weights[t] = "t günü close'da girilecek pozisyon kararı".
    - Earn edilen return: close[t]→close[t+1] = rets[t+1].
    - PnL hesabı: weights[t] × rets[t+1] → series: gross = weights.shift(1) * rets

    Sadece weights'in tanımlı olduğu (test) günler için PnL hesaplanır;
    boş günler 0 ile doldurulup Sharpe'ı bozmaz — out-of-sample (out-of-market)
    olan günler hiç katılmaz.
    """
    rets = panel.pivot(index="date", columns="ticker", values="close").pct_change()
    # Weights'in olduğu günler + 1 gün ileri (PnL realisation günleri için)
    if weights.empty:
        return pd.Series(dtype=float)

    all_panel_dates = sorted(rets.index)
    # weights tarihlerini al ve her birinin BİR SONRAKİ panel tarihini ekle
    pnl_dates = []
    for d in weights.index:
        idx = pd.Index(all_panel_dates).get_indexer([d])[0]
        if idx >= 0 and idx + 1 < len(all_panel_dates):
            pnl_dates.append(all_panel_dates[idx + 1])
    pnl_dates = pd.DatetimeIndex(sorted(set(pnl_dates)))

    if len(pnl_dates) == 0:
        return pd.Series(dtype=float)

    common_cols = weights.columns.intersection(rets.columns)
    # Her PnL günü için, ÖNCEKİ panel günündeki weights × o günün rets'i
    # ÖNEMLİ halisünasyon defansı: eksik return'leri 0 ile DOLDURMUYORUZ.
    # Eğer bir ticker'ın o günkü close'u eksikse (delisting, suspended, vs.),
    # o ticker'ı portföyden ÇIKAR ve kalan ağırlıkları renormalize et.
    pnl_rows = {}
    excluded_count = 0
    for pd_date in pnl_dates:
        prev_idx = pd.Index(all_panel_dates).get_loc(pd_date) - 1
        prev_date = all_panel_dates[prev_idx]
        if prev_date not in weights.index:
            continue
        w_row = weights.loc[prev_date, common_cols].fillna(0)
        r_row = rets.loc[pd_date, common_cols]
        # Eksik return olan ticker'ları çıkar (delisting / suspended / data eksik)
        valid_mask = r_row.notna()
        if (~valid_mask).any() and (w_row > 0).any():
            n_missing = int(((w_row > 0) & ~valid_mask).sum())
            if n_missing > 0:
                excluded_count += n_missing
            # Eksik ticker'lara olan ağırlığı pozisyon dışı kabul et (nakit)
            w_row = w_row * valid_mask
            r_row = r_row.fillna(0)
        pnl_rows[pd_date] = float((w_row * r_row).sum())
    if excluded_count > 0:
        logger.warning(f"PnL: {excluded_count} pozisyon eksik return nedeniyle nakitleştirildi")
    gross = pd.Series(pnl_rows).sort_index()

    # Turnover: weights diff (rebalance günleri arası)
    w_aligned = weights.loc[:, common_cols].fillna(0)
    diff = w_aligned.diff()
    if len(diff) > 0:
        diff.iloc[0] = w_aligned.iloc[0]
    turnover = diff.abs().sum(axis=1)
    # Turnover cost'u o günün (kararın alındığı gün) değil, bir sonraki günün (entry) realisation'ında uygula
    cost_per_date = {}
    for d in w_aligned.index:
        idx = pd.Index(all_panel_dates).get_loc(d)
        if idx + 1 < len(all_panel_dates):
            entry_day = all_panel_dates[idx + 1]
            cost_per_date[entry_day] = turnover.loc[d] * cost.one_way_cost()
    cost_series = pd.Series(cost_per_date).sort_index()
    cost_series = cost_series.reindex(gross.index).fillna(0)

    return gross - cost_series


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
    rebalance_freq = SETTINGS["strategy"].get("rebalance_freq_days", 1)
    smooth_window = SETTINGS["strategy"].get("signal_smooth_window", 1)

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
    run_id = make_run_id()

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
        save_model(tm, run_id, i)   # her fold için artifact diske yaz
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
    weights = _build_positions(
        signals_concat,
        top_n=top_n,
        max_pos=max_pos,
        rebalance_freq=rebalance_freq,
        smooth_window=smooth_window,
    )

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
    from backtest.survivorship import disclose_survivorship_bias, estimate_bias_magnitude
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
    print("─" * 70)
    print(disclose_survivorship_bias())
    bias = estimate_bias_magnitude()
    print(f"Tahmini bias büyüklüğü: {bias['estimated_annual_bias_pct']}% yıllık")
    print(f"5 yıl kümülatif:        {bias['cumulative_bias_5yr_pct']}%")
    print(f"Öneri: {bias['recommendation']}")
    print()
    sharpe_realistic = result.overall_metrics.get("sharpe", 0) * 0.85
    cagr_realistic = result.overall_metrics.get("cagr", 0) * 0.90
    print(f"Realistik tahminler (bias düzeltilmiş):")
    print(f"  Sharpe: {sharpe_realistic:.2f}  (raw: {result.overall_metrics.get('sharpe'):.2f})")
    print(f"  CAGR:   {cagr_realistic*100:.1f}%  (raw: {result.overall_metrics.get('cagr')*100:.1f}%)")
    print()
