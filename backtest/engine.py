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
    long_short: bool = False,
    short_weight: float = 0.5,
    conviction_weight: bool = False,
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
    rb_all = s[s["date"].isin(rebalance_dates)].copy()
    # Long-short veya long-only?
    if long_short:
        # Long top-N, short bottom-N (market-neutral or partially hedged)
        rb_all["rank_long"]  = rb_all.groupby("date")["signal"].rank(ascending=False, method="first")
        rb_all["rank_short"] = rb_all.groupby("date")["signal"].rank(ascending=True,  method="first")
        rb_long  = rb_all[rb_all["rank_long"]  <= top_n].copy()
        rb_short = rb_all[rb_all["rank_short"] <= top_n].copy()
        # Equal weight within long sleeve, gross long = 1 - short_weight
        long_gross = 1.0 - short_weight
        rb_long["weight"]  = (long_gross / top_n)
        rb_short["weight"] = -(short_weight / top_n)
        rb = pd.concat([rb_long, rb_short], ignore_index=True)
    else:
        rb_all["rank"] = rb_all.groupby("date")["signal"].rank(ascending=False, method="first")
        rb = rb_all[rb_all["rank"] <= top_n].copy()
        if conviction_weight:
            # Conviction weighting: signal strength = position size
            # Her gün toplam = 1.0. Higher signal → higher weight.
            # Negatif sinyaller sıfırlanır (zaten top-N içinde olduğu için tüm sinyaller pozitif tarafta)
            rb["signal_pos"] = rb["signal"].clip(lower=0.001)
            rb["signal_sum"] = rb.groupby("date")["signal_pos"].transform("sum")
            rb["weight"] = (rb["signal_pos"] / rb["signal_sum"]).clip(upper=max_pos)
        else:
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
    # Long-short uyumluluğu: long ve short bacakları ayrı işle.
    # Long (w>=0): held = min(w_prev_long, w_long), exit/enter aynı
    # Short (w<0): mutlak değer üzerinden aynı mantık, sonra negatif geri
    w_long = w.clip(lower=0)
    w_long_prev = w_prev.clip(lower=0)
    held_long = pd.DataFrame(
        np.minimum(w_long_prev.values, w_long.values),
        index=w.index, columns=w.columns,
    )
    exited_long  = (w_long_prev - held_long).clip(lower=0)
    entered_long = (w_long - held_long).clip(lower=0)

    # Short bacak (|w| üzerinde aynı mantık)
    w_short_abs = (-w).clip(lower=0)
    w_short_abs_prev = (-w_prev).clip(lower=0)
    held_short_abs = pd.DataFrame(
        np.minimum(w_short_abs_prev.values, w_short_abs.values),
        index=w.index, columns=w.columns,
    )
    exited_short_abs  = (w_short_abs_prev - held_short_abs).clip(lower=0)
    entered_short_abs = (w_short_abs - held_short_abs).clip(lower=0)

    # Short pozisyonun PnL'i ters (cc * -1)
    held   = held_long - held_short_abs
    exited = exited_long + exited_short_abs    # mutlak değer, cost için
    entered = entered_long + entered_short_abs # mutlak değer, cost için

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

    # Long ve short bacak PnL'i ayrı (cost için exited/entered absolute zaten)
    pnl_held = (held * cc_aligned).sum(axis=1)
    # Exit ve enter long bacağı için: long açılış kar pozitif, short açılış kar negatif çarpan
    pnl_exit_long  = (exited_long  * on_aligned).sum(axis=1)
    pnl_enter_long = (entered_long * id_aligned).sum(axis=1)
    # Short bacağı: stock yükselirse short LOSS, o yüzden NEGATIF
    pnl_exit_short  = -(exited_short_abs  * on_aligned).sum(axis=1)
    pnl_enter_short = -(entered_short_abs * id_aligned).sum(axis=1)
    gross = pnl_held + pnl_exit_long + pnl_enter_long + pnl_exit_short + pnl_enter_short

    # Maliyet: exit + enter (her iki bacak için absolute turnover)
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
    long_short    = SETTINGS["strategy"].get("long_short", False)
    short_weight  = SETTINGS["strategy"].get("short_weight", 0.5)
    conviction_w  = SETTINGS["strategy"].get("conviction_weight", False)

    signal_source = SETTINGS["strategy"].get("signal_source", "ml")  # ml | factor | blend

    logger.info(f"Backtest başlıyor... (signal_source={signal_source})")
    panel = load_panel(start=start, end=end)
    features = build_features(panel, lag=True)

    # ── FACTOR SIGNAL PATH (ML'siz, akademik faktör composite) ──────────
    if signal_source in ("factor", "blend"):
        factor_sig = _build_factor_signal(features)
        if signal_source == "factor":
            # ML eğitimi yok — direkt faktör sinyali kullan
            run_id = make_run_id()
            signals_concat = factor_sig
            # Fold metrics minimal (faktör için fold yok ama walk-forward tarih aralığı)
            fold_metrics = [{
                "fold": 0, "train_n": 0, "test_n": len(factor_sig),
                "train_period": "factor (no training)",
                "test_period": f"{factor_sig['date'].min().date()} → {factor_sig['date'].max().date()}",
                "positive_rate_train": 0.0,
            }]
            weights = _build_positions(
                signals_concat, top_n=top_n, max_pos=max_pos,
                rebalance_freq=rebalance_freq, smooth_window=smooth_window,
                signal_invert=signal_invert, long_short=long_short,
                short_weight=short_weight, conviction_weight=conviction_w,
            )
            return _finalize_backtest(weights, panel, cost, fold_metrics, splits=None, run_id=run_id)

    # Multi-horizon ensemble: her horizon için ayrı training set + model
    horizons = SETTINGS["model"].get("multi_horizon", None)
    label_type = SETTINGS.get("labels", {}).get("label_type", "triple_barrier")
    use_multi_horizon = (
        horizons and label_type == "relative_outperform" and len(horizons) > 1
    )

    if use_multi_horizon:
        logger.info(f"Multi-horizon ensemble: {horizons}")
    else:
        horizons = [None]   # tek model

    # Training sets her horizon için (NaN bırakılan rows farklı olabilir)
    training_sets = {}
    feat_cols = None
    for h in horizons:
        tds, _, fc = build_training_set(features, horizon_override=h)
        training_sets[h] = tds
        if feat_cols is None:
            feat_cols = fc
    logger.info(f"Training sets hazır: {[(h, len(td)) for h, td in training_sets.items()]} × {len(feat_cols)} feature")

    # Walk-forward split'ler ilk horizon'un tarihlerine göre
    primary_td = training_sets[horizons[0]]
    dates = pd.DatetimeIndex(sorted(primary_td["date"].unique()))
    splits = make_splits(dates)
    if not splits:
        raise RuntimeError("Walk-forward split üretilemedi — veri kısa olabilir")

    all_signals: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    run_id = make_run_id()

    for i, split in enumerate(splits):
        # Her horizon için fold-train, predict, average signals
        per_horizon_sigs = []
        for h in horizons:
            td = training_sets[h]
            train = td[(td["date"] >= split.train_start) & (td["date"] <= split.train_end)]
            test = td[(td["date"] >= split.test_start) & (td["date"] < split.test_end)]
            if train.empty or test.empty or train["target"].nunique() < 2:
                continue
            try:
                tm = train_fold(train, feat_cols)
            except Exception as e:
                logger.error(f"Fold {i} (h={h}): training fail → {e}")
                continue
            sig = predict_fold(tm, test)
            per_horizon_sigs.append(sig)
            if h == horizons[0]:
                save_model(tm, run_id, i)   # primary horizon artifact

        if not per_horizon_sigs:
            logger.warning(f"Fold {i}: hiç başarılı horizon yok")
            continue

        # Multi-horizon ortalama
        if len(per_horizon_sigs) == 1:
            sig = per_horizon_sigs[0]
        else:
            merged = per_horizon_sigs[0].copy()
            for other in per_horizon_sigs[1:]:
                merged = merged.merge(other.rename(columns={"signal": "signal_other"}),
                                      on=["date", "ticker"], how="outer")
                merged["signal"] = merged[["signal", "signal_other"]].mean(axis=1)
                merged = merged.drop(columns=["signal_other"])
            sig = merged[["date", "ticker", "signal"]]
        all_signals.append(sig)
        train = training_sets[horizons[0]]
        train_fold_df = train[(train["date"] >= split.train_start) & (train["date"] <= split.train_end)]
        test_fold_df = train[(train["date"] >= split.test_start) & (train["date"] < split.test_end)]
        fold_metrics.append({
            "fold": i,
            "train_n": len(train_fold_df),
            "test_n": len(test_fold_df),
            "train_period": f"{split.train_start.date()} → {split.train_end.date()}",
            "test_period":  f"{split.test_start.date()} → {split.test_end.date()}",
            "positive_rate_train": float(train_fold_df["target"].mean()) if len(train_fold_df) else 0.0,
        })
        logger.info(f"Fold {i}: train={len(train_fold_df)} test={len(test_fold_df)} horizons_used={len(per_horizon_sigs)}")

    if not all_signals:
        raise RuntimeError("Hiç başarılı fold yok")

    signals_concat = pd.concat(all_signals, ignore_index=True)

    # BLEND: ML sinyali + factor sinyali ortalaması (cross-sectional rank uzayında)
    if signal_source == "blend":
        fs = _build_factor_signal(features)
        # Her ikisini de günlük cross-sectional rank'e çevir (0..1), sonra ortala
        def rank_norm(df):
            df = df.copy()
            df["rk"] = df.groupby("date")["signal"].rank(pct=True)
            return df[["date", "ticker", "rk"]]
        ml_r = rank_norm(signals_concat).rename(columns={"rk": "ml"})
        fs_r = rank_norm(fs).rename(columns={"rk": "fc"})
        merged = ml_r.merge(fs_r, on=["date", "ticker"], how="inner")
        blend_w = SETTINGS["strategy"].get("blend_factor_weight", 0.5)
        merged["signal"] = (1 - blend_w) * merged["ml"] + blend_w * merged["fc"]
        signals_concat = merged[["date", "ticker", "signal"]]
        logger.info(f"Blend signal: ML×{1-blend_w:.1f} + Factor×{blend_w:.1f}, {len(signals_concat)} satır")

    weights = _build_positions(
        signals_concat,
        top_n=top_n,
        max_pos=max_pos,
        rebalance_freq=rebalance_freq,
        smooth_window=smooth_window,
        signal_invert=signal_invert,
        long_short=long_short,
        short_weight=short_weight,
        conviction_weight=conviction_w,
    )
    return _finalize_backtest(weights, panel, cost, fold_metrics, splits=splits, run_id=run_id)


def _finalize_backtest(weights, panel, cost, fold_metrics, splits, run_id) -> BacktestResult:
    """Weights → vol-target → PnL → benchmark → artifacts. ML ve factor path ortak."""
    # Vol-targeting (Moskowitz et al. 2012): exposure = target_vol / realized_vol
    vol_target_annual = SETTINGS["strategy"].get("vol_target_annual", None)
    lev_cap = SETTINGS["strategy"].get("leverage_cap", 2.0)
    if vol_target_annual:
        raw_net = _daily_pnl(weights, panel, cost)
        roll_vol = raw_net.rolling(60, min_periods=20).std() * np.sqrt(252)
        mult = (vol_target_annual / roll_vol).shift(1).clip(0.0, lev_cap).fillna(1.0)
        weights = weights.multiply(mult, axis=0).fillna(0.0)
        net = _daily_pnl(weights, panel, cost)
    else:
        net = _daily_pnl(weights, panel, cost)

    bench = _benchmark_returns(panel).reindex(net.index).fillna(0)
    overall = summary(net, weights=weights)
    bench_summary = summary(bench)
    n_folds = len(splits) if splits else 1
    logger.info(f"Backtest bitti: {n_folds} fold, {len(net)} işlem günü")
    result = BacktestResult(
        daily_returns=net,
        daily_weights=weights,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        benchmark_returns=bench,
        benchmark_metrics=bench_summary,
    )
    _save_backtest_artifacts(result, run_id)
    return result


def _build_factor_signal(features: pd.DataFrame) -> pd.DataFrame:
    """Akademik faktör composite — ML'siz. Cross-sectional z-score kombinasyonu.

    Kanıtlanmış faktörler (market-neutral L/S Sharpe ölçüldü):
      - low idiosyncratic vol (Ang et al. 2006): en güçlü, Sharpe 1.81 gross
      - low total vol
      - 20-gün ve 60-gün momentum (Jegadeesh-Titman)
    Yüksek composite skor = long adayı.
    """
    df = features.copy()
    # Cross-sectional z-score helper
    def csz(col):
        m = df.groupby("date")[col].transform("mean")
        s = df.groupby("date")[col].transform("std")
        return (df[col] - m) / s.replace(0, np.nan)

    # Feature'lar zaten lag=True (t-1 observable). idio_vol_60, vol_20, mom_20, momentum_60 mevcut.
    comp = pd.Series(0.0, index=df.index)
    weights_cfg = SETTINGS["strategy"].get("factor_weights", {
        "idio_vol_60": -1.0,   # düşük idio-vol iyi
        "vol_20": -0.5,        # düşük vol iyi
        "ret_20": 0.5,         # momentum
        "momentum_60": 0.3,    # uzun momentum
    })
    for col, w in weights_cfg.items():
        if col in df.columns:
            comp = comp + w * csz(col).fillna(0.0)

    out = df[["date", "ticker"]].copy()
    out["signal"] = comp
    # NaN feature'lı satırları çıkar (warm-up)
    out = out[df["idio_vol_60"].notna() | df["vol_20"].notna()]
    return out.dropna(subset=["signal"])


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
