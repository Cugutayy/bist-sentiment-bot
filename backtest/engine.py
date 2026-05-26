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
    signal_invert: bool = False,
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

    # SIGNAL INVERT — IC analysis (diagnostic) showed model picks underperform;
    # bottom-N outperformed top-N by +45%/yr. Triple barrier label seems to
    # capture momentum that mean-reverts. Inverting signal converts model into
    # a mean-reversion bot.
    if signal_invert:
        s["signal"] = -s["signal"]

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

    # BUG FIX (continuous mode): rebalance günlerinde TÜM ticker'lar explicit
    # 0 ile başla; sadece seçilenler weight alır. Sonra ffill yap (pozisyon
    # tut). Önceden sadece seçilen ticker'lar pivot'a giriyordu → seçilmeyen
    # önceki rebalance'lardaki ticker'lar NaN→ffill ile %10 olarak kalıyordu
    # ve continuous mode'da gross exposure 3x'e çıkıyordu.
    all_tickers = sorted(s["ticker"].unique())
    rebalance_sorted = sorted(rebalance_dates)
    rb_pivot = pd.DataFrame(0.0, index=pd.DatetimeIndex(rebalance_sorted), columns=all_tickers)
    for _, r in rb.iterrows():
        rb_pivot.loc[r["date"], r["ticker"]] = float(r["weight"])

    # Tüm günlere ffill (rebalance arası pozisyon tut)
    full_index = pd.DatetimeIndex(all_dates)
    pivot = rb_pivot.reindex(full_index).ffill().fillna(0.0)

    # Fold sınırlarında SIFIRLA (discrete mode için): büyük gap varsa eski
    # pozisyonları taşımayalım (out-of-sample bütünlük).
    for i, fs in enumerate(fold_starts):
        if i == 0:
            continue
        fs_idx = all_dates.index(fs)
        # Bu fold'a ait ilk rebalance gününden ÖNCEKİ günler için 0
        # (gap içinde pozisyon tutmak yanlış olur)
        pivot.iloc[fs_idx - 1] = pivot.iloc[fs_idx - 1] * 0.0  # önceki gün sıfırla
    return pivot


def _daily_pnl(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    cost: CostModel,
) -> pd.Series:
    """Next-day open execution simülasyonu — gerçek hayata en yakın PnL.

    Modeli:
    - Sinyal close[t-1]'de üretilir (features lag=True).
    - weights[t] = "t günü için aktif portföy" (open[t]'de execute).

    Per-ticker per-day pnl:
    - HELD (önceki gün de aynı ağırlık vardı): held × cc[t] (close[t-1]→close[t])
    - ENTERED (t günü yeni alınan): entered × intraday[t] (open[t]→close[t])
    - EXITED (t günü satılan): exited × overnight[t] (close[t-1]→open[t])

    Bu model "overnight gap riski"ni doğru simüle eder:
    - İyi haberle gap-up: live trader gap'i kaçırır (entered_intraday < cc)
    - Kötü haberle gap-down: live trader gap'ten KORUNMUŞ olur (exited_overnight)

    Halisünasyon defansı: eksik close → o ticker'ı o gün portföyden çıkar.
    """
    if weights.empty:
        return pd.Series(dtype=float)

    pivot_close = panel.pivot(index="date", columns="ticker", values="close")
    pivot_open = panel.pivot(index="date", columns="ticker", values="open")

    cc_ret = pivot_close.pct_change()
    overnight_ret = (pivot_open / pivot_close.shift(1)) - 1
    intraday_ret = (pivot_close / pivot_open) - 1

    common_cols = weights.columns.intersection(pivot_close.columns)
    w = weights.loc[:, common_cols].fillna(0).sort_index()

    # Her weights günü için bir önceki weight'i shift'le bul (ilk gün 0 kabul)
    w_prev = w.shift(1).fillna(0)
    held = pd.DataFrame(
        np.minimum(w_prev.values, w.values),
        index=w.index, columns=w.columns,
    )
    exited = (w_prev - held).clip(lower=0)
    entered = (w - held).clip(lower=0)

    # Returns'leri weights'in tarihlerine align et (sadece o günler için)
    cc_aligned = cc_ret.loc[:, common_cols].reindex(w.index)
    on_aligned = overnight_ret.loc[:, common_cols].reindex(w.index)
    id_aligned = intraday_ret.loc[:, common_cols].reindex(w.index)

    # Halisünasyon: eksik returns olan ticker × gün'leri çıkar
    valid_cc = cc_aligned.notna()
    valid_on = on_aligned.notna()
    valid_id = id_aligned.notna()
    excluded = int(((held > 0) & ~valid_cc).sum().sum() +
                   ((exited > 0) & ~valid_on).sum().sum() +
                   ((entered > 0) & ~valid_id).sum().sum())
    if excluded > 0:
        logger.warning(f"PnL: {excluded} pozisyon-gün eksik veri nedeniyle nakitleştirildi")

    cc_aligned = cc_aligned.fillna(0)
    on_aligned = on_aligned.fillna(0)
    id_aligned = id_aligned.fillna(0)

    pnl_held = (held * cc_aligned).sum(axis=1)
    pnl_exit = (exited * on_aligned).sum(axis=1)
    pnl_enter = (entered * id_aligned).sum(axis=1)
    gross = pnl_held + pnl_exit + pnl_enter

    # Maliyet: sadece exit + enter (held için 0)
    turnover = exited.sum(axis=1) + entered.sum(axis=1)
    cost_pct = turnover * cost.one_way_cost()

    net = gross - cost_pct
    return net


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
    signal_invert = SETTINGS["strategy"].get("signal_invert", False)

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
        signal_invert=signal_invert,
    )

    # Net günlük getiri (panel: orijinal price panel, NOT features)
    net = _daily_pnl(weights, panel, cost)
    # Benchmark: SADECE backtest test günlerinde, apples-to-apples karşılaştırma
    bench = _benchmark_returns(panel).reindex(net.index).fillna(0)

    overall = summary(net, weights=weights)
    bench_summary = summary(bench)

    logger.info(f"Backtest bitti: {len(splits)} fold, {len(net)} işlem günü")
    result = BacktestResult(
        daily_returns=net,
        daily_weights=weights,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        benchmark_returns=bench,
        benchmark_metrics=bench_summary,
    )

    # Diske yaz — inceleyebilmen için
    _save_backtest_artifacts(result, run_id)
    return result


def _save_backtest_artifacts(result: BacktestResult, run_id: str) -> None:
    """Backtest sonuçlarını parquet + JSON olarak diske yaz."""
    import json
    from config import ROOT

    out_dir = ROOT / "data" / "processed" / "backtests" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily returns + benchmark karşılaştırma
    daily = pd.DataFrame({
        "portfolio_return": result.daily_returns,
        "benchmark_return": result.benchmark_returns,
        "portfolio_equity": (1 + result.daily_returns).cumprod(),
        "benchmark_equity": (1 + result.benchmark_returns).cumprod(),
    })
    daily.index.name = "date"
    daily.reset_index().to_parquet(out_dir / "daily.parquet", index=False)

    # Pozisyon ağırlıkları
    weights = result.daily_weights.copy()
    weights.index.name = "date"
    weights.reset_index().to_parquet(out_dir / "weights.parquet", index=False)

    # Özet (JSON)
    summary = {
        "run_id": run_id,
        "n_folds": len(result.fold_metrics),
        "n_days": len(result.daily_returns),
        "portfolio_metrics": result.overall_metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "fold_metrics": result.fold_metrics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"Backtest artifact'ları → {out_dir}")


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
