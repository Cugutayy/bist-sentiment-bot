"""LOOP ITERATION 3 — Block bootstrap robust proof.

Iter 2'de monthly-HAC p=0.0419 ile <0.05 geciyoruz, ama tek t-test sonucu
dagilim varsayimi tasiyor. Bootstrap ile non-parametric kanit:

1. STATIONARY BLOCK BOOTSTRAP: otokorelasyon yapisini koruyarak gunluk
   getirileri yeniden orneklenme. 5000 replikasyon -> percentile CI.

2. SHARPE RATIO BOOTSTRAP: Sharpe(strategy) - Sharpe(SPY) > 0 oldugunu
   bootstrap CI ile gosterir.

3. ALPHA HARITASI: rolling 1-yillik excess Sharpe'in zamanda istikrarli
   pozitif olup olmadigi (single-period luck mu yapisal mi?).

Proof: bootstrap CI 95% pozitif olursa kanit saglam.
"""
from __future__ import annotations
import sys, math, warnings, io
warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices
from trend.strategy import run_trend_backtest, TrendConfig, DEFAULT_UNIVERSE

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD",
          "AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]
ANN = 252
RF  = 0.04

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

def stationary_block_bootstrap(data, n_boot=5000, mean_block=10, seed=42):
    """Politis-Romano stationary block bootstrap: bloklar GEO dagilimli."""
    rng = np.random.default_rng(seed)
    n = len(data)
    p_geom = 1.0 / mean_block
    boots = np.empty((n_boot, n))
    arr = np.asarray(data)
    for b in range(n_boot):
        out = np.empty(n)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            L = rng.geometric(p_geom)
            L = min(L, n - i)
            for k in range(L):
                out[i+k] = arr[(start + k) % n]
            i += L
        boots[b] = out
    return boots

def main():
    print("="*70)
    print("LOOP ITERATION 3 — BOOTSTRAP ROBUST PROOF")
    print("="*70)

    print("\n[1/4] Loading + building strategies...")
    crypto_close = fetch_prices(CRYPTO, period="5y")
    multi_close  = fetch_prices(DEFAULT_UNIVERSE, period="10y")
    crypto_r = crypto_ema_sleeve(crypto_close)
    multi_r = run_trend_backtest(multi_close, TrendConfig(vol_target_annual=0.20)).daily_returns
    common = crypto_r.index.intersection(multi_r.index)
    combined = (0.5*crypto_r.reindex(common) + 0.5*multi_r.reindex(common))
    spy = multi_close["SPY"].pct_change().reindex(common).dropna()
    combined = combined.reindex(spy.index).dropna()
    spy = spy.reindex(combined.index)
    excess = (combined - spy).dropna()
    print(f"  n = {len(excess)} days, period {excess.index[0].date()} -> {excess.index[-1].date()}")

    # ---------- Bootstrap 1: mean excess return ----------
    print("\n[2/4] Bootstrap mean excess return (5000 reps, block bootstrap)...")
    boots = stationary_block_bootstrap(excess.values, n_boot=5000, mean_block=10)
    boot_means = boots.mean(axis=1)
    obs_mean = excess.mean()
    p_boot = (boot_means <= 0).mean()  # P(boot mean <= 0) under perm
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"  Observed mean excess: {obs_mean*1e4:+.2f} bps/day  ({(1+obs_mean)**252-1:+.2%}/yr)")
    print(f"  Bootstrap 95% CI: [{ci_low*1e4:+.2f}, {ci_high*1e4:+.2f}] bps/day")
    print(f"  P(mean <= 0): {p_boot:.4f}  ({'PASS' if p_boot < 0.05 else 'FAIL'} at 5%)")

    # ---------- Bootstrap 2: Sharpe ratio diff ----------
    print("\n[3/4] Bootstrap Sharpe diff (strategy − SPY)...")
    def sh(r, ann=252):
        r = np.asarray(r)
        m = r.mean()*ann - RF
        s = r.std(ddof=1)*math.sqrt(ann)
        return m/s if s > 0 else 0
    # Pair-wise bootstrap (combined, spy)
    rng = np.random.default_rng(43)
    n = len(combined)
    sh_obs_diff = sh(combined.values) - sh(spy.values)
    sh_diffs = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, n, n)
        sh_diffs[b] = sh(combined.values[idx]) - sh(spy.values[idx])
    sh_ci_low, sh_ci_high = np.percentile(sh_diffs, [2.5, 97.5])
    sh_p = (sh_diffs <= 0).mean()
    print(f"  Observed Sharpe diff: {sh_obs_diff:+.4f}")
    print(f"  Bootstrap 95% CI: [{sh_ci_low:+.4f}, {sh_ci_high:+.4f}]")
    print(f"  P(diff <= 0): {sh_p:.4f}  ({'PASS' if sh_p < 0.05 else 'FAIL'} at 5%)")

    # ---------- Rolling 252-day excess Sharpe ----------
    print("\n[4/4] Rolling 1-yr excess Sharpe (stability test)...")
    roll_mean = excess.rolling(252).mean()
    roll_std = excess.rolling(252).std()
    roll_sh = (roll_mean / roll_std * math.sqrt(252)).dropna()
    pct_positive = (roll_sh > 0).mean()
    print(f"  Mean rolling 1y excess Sharpe: {roll_sh.mean():+.3f}")
    print(f"  Min / Median / Max: {roll_sh.min():+.3f} / {roll_sh.median():+.3f} / {roll_sh.max():+.3f}")
    print(f"  % time positive: {pct_positive*100:.1f}%")
    print(f"  Stability: {'STABLE (>=70% positive)' if pct_positive >= 0.70 else 'UNSTABLE'}")

    # PROOF
    proven = (p_boot < 0.05) and (sh_p < 0.05) and (pct_positive >= 0.70)
    print(f"\n  ROBUST PROOF:")
    print(f"    Mean excess > 0      : {'PASS' if p_boot < 0.05 else 'FAIL'}  (p_boot={p_boot:.4f})")
    print(f"    Sharpe diff > 0      : {'PASS' if sh_p < 0.05 else 'FAIL'}  (p_boot={sh_p:.4f})")
    print(f"    Rolling 1y stability : {'PASS' if pct_positive >= 0.70 else 'FAIL'}  ({pct_positive*100:.1f}%)")
    print(f"\n  RESULT: {'ROBUST PROOF — strategy beats SPY in 3 independent tests' if proven else 'PARTIAL PROOF — refine'}")
    return proven, p_boot, sh_p, pct_positive

if __name__ == "__main__":
    main()
