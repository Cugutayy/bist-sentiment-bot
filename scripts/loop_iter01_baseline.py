"""LOOP ITERATION 1 — baseline measurement + statistical proof test.

Mevcut en iyi sistemin (kombine: kripto EMA3/15 + multi-asset trend blend)
SPY'i istatistiksel olarak gecip gecmedigini test et.

Proof kriterleri:
  (a) CAGR > SPY CAGR
  (b) Sharpe > SPY Sharpe
  (c) Excess return t-test: H0: mu_excess <= 0 vs H1: mu_excess > 0, p < 0.05

Sonuc:
  - Tum 3 kriter geciyorsa: KANIT EDILDI, loop son.
  - Bir veya birden fazla kriter gecemiyorsa: HIPOTEZ ile bir sonraki iterasyon.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import math
from trend.data import fetch_prices
from trend.strategy import run_trend_backtest, TrendConfig, DEFAULT_UNIVERSE

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD",
          "AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]
ANN = 252
RF = 0.04  # USD risk-free

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def crypto_ema_sleeve(close, fast=3, slow=15, vol_target=0.30, ann=365, cost=0.0010):
    rets = close.pct_change()
    sig = np.sign(close.apply(lambda c: ema(c,fast)) - close.apply(lambda c: ema(c,slow))).clip(lower=0).shift(1)
    cvol = rets.rolling(30).std()*np.sqrt(ann)
    pos = sig*(vol_target/cvol).clip(0,3).shift(1)/close.shape[1]
    raw = (pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost
    raw = raw.dropna()
    realized = raw.rolling(60,min_periods=20).std()*np.sqrt(ann)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

def annualize_metrics(daily, rf=RF, ann=ANN):
    r = daily.dropna()
    cagr = (1+r).prod()**(ann/len(r)) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean()*ann - rf) / vol if vol > 0 else 0
    eq = (1+r).cumprod()
    dd = ((eq/eq.cummax())-1).min()
    return cagr, sharpe, dd, vol

def excess_t_test(strategy, bench, ann=ANN):
    """One-sided t-test: H0: mu_excess <= 0, H1: mu_excess > 0."""
    common = strategy.index.intersection(bench.index)
    s = strategy.reindex(common).dropna()
    b = bench.reindex(common).dropna()
    common = s.index.intersection(b.index)
    excess = (s.reindex(common) - b.reindex(common)).dropna()
    n = len(excess)
    mean = excess.mean()
    sd = excess.std()
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else 0
    df = n - 1
    # one-sided p (approximation using normal distribution for large df)
    # use t-distribution via scipy if available; else normal approx
    try:
        from scipy import stats as sstats
        p = 1 - sstats.t.cdf(t, df)
    except ImportError:
        # normal approximation
        from math import erf
        p = 1 - 0.5*(1 + erf(t/math.sqrt(2)))
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t, "df": df, "p_one_sided": p,
            "ann_excess": (1+mean)**ann - 1}

def main():
    print("="*70)
    print("LOOP ITERATION 1 — BASELINE STATISTICAL PROOF TEST")
    print("="*70)

    print("\n[1/3] Loading data...")
    crypto_close = fetch_prices(CRYPTO, period="5y")
    multi_close  = fetch_prices(DEFAULT_UNIVERSE, period="10y")
    print(f"  Crypto: {crypto_close.shape[1]} coins, {len(crypto_close)} days")
    print(f"  Multi:  {multi_close.shape[1]} assets, {len(multi_close)} days")

    print("\n[2/3] Building strategies...")
    crypto_r = crypto_ema_sleeve(crypto_close)
    multi_res = run_trend_backtest(multi_close, TrendConfig(vol_target_annual=0.20))
    multi_r = multi_res.daily_returns
    # Kombine 50/50 — multi-asset takvimine ortala (multi 252, crypto 365 -> ortak)
    common_idx = crypto_r.index.intersection(multi_r.index)
    cr = crypto_r.reindex(common_idx); mr = multi_r.reindex(common_idx)
    combined = 0.5*cr + 0.5*mr

    # SPY benchmark (multi'nin universe'unda)
    spy = multi_close["SPY"].pct_change().reindex(common_idx).dropna()
    combined = combined.reindex(spy.index).dropna()

    print(f"  Common period: {combined.index[0].date()} -> {combined.index[-1].date()}  ({len(combined)} days)")

    print("\n[3/3] Computing metrics & t-test...")
    s_cagr, s_sharpe, s_dd, s_vol = annualize_metrics(combined)
    b_cagr, b_sharpe, b_dd, b_vol = annualize_metrics(spy)

    print(f"\n  STRATEGY (Kombine 50/50)")
    print(f"    CAGR:   {s_cagr*100:+7.2f}%")
    print(f"    Sharpe: {s_sharpe:+7.3f}")
    print(f"    Vol:    {s_vol*100:7.2f}%")
    print(f"    MaxDD:  {s_dd*100:+7.1f}%")

    print(f"\n  BENCHMARK (SPY)")
    print(f"    CAGR:   {b_cagr*100:+7.2f}%")
    print(f"    Sharpe: {b_sharpe:+7.3f}")
    print(f"    Vol:    {b_vol*100:7.2f}%")
    print(f"    MaxDD:  {b_dd*100:+7.1f}%")

    tt = excess_t_test(combined, spy)
    print(f"\n  EXCESS RETURN t-TEST (H0: mu_excess <= 0)")
    print(f"    n:           {tt['n']}")
    print(f"    mean excess: {tt['mean']*1e4:.2f} bps/day  ({tt['ann_excess']*100:+.2f}%/year)")
    print(f"    sd:          {tt['sd']*100:.4f}%")
    print(f"    t-stat:      {tt['t']:+.4f}")
    print(f"    p (1-sided): {tt['p_one_sided']:.4f}")

    # Proof check
    crit_cagr = s_cagr > b_cagr
    crit_sharpe = s_sharpe > b_sharpe
    crit_ttest = tt['p_one_sided'] < 0.05 and tt['t'] > 0

    print(f"\n  PROOF CRITERIA:")
    print(f"    (a) CAGR > SPY      : {'PASS' if crit_cagr else 'FAIL'}  ({s_cagr*100:.2f}% vs {b_cagr*100:.2f}%)")
    print(f"    (b) Sharpe > SPY    : {'PASS' if crit_sharpe else 'FAIL'}  ({s_sharpe:.3f} vs {b_sharpe:.3f})")
    print(f"    (c) Excess t p<0.05 : {'PASS' if crit_ttest else 'FAIL'}  (p={tt['p_one_sided']:.4f})")

    proven = crit_cagr and crit_sharpe and crit_ttest
    print(f"\n  RESULT: {'PROVEN — strategy beats SPY' if proven else 'NOT PROVEN — continue loop'}")

    if not proven:
        print("\n  WEAKNESS ANALYSIS:")
        if not crit_cagr:
            print(f"    - CAGR shortfall: {(b_cagr-s_cagr)*100:.2f}pp")
        if not crit_sharpe:
            print(f"    - Sharpe shortfall: {b_sharpe-s_sharpe:.3f}")
        if not crit_ttest:
            print(f"    - Excess t-test insignificant: p={tt['p_one_sided']:.4f}")
            if tt['t'] > 0:
                print(f"      Positive excess but not significant (n={tt['n']}, power issue)")
            else:
                print(f"      Negative excess (strategy underperforms)")
    return proven, s_sharpe, b_sharpe, s_cagr, b_cagr

if __name__ == "__main__":
    main()
