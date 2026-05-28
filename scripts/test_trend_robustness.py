"""Trend stratejisi OVERFIT kontrolu — parametre robustlugu + OOS.

Overfit testi: Sharpe sadece TEK parametre kombosunda mi pozitif (overfit),
yoksa GENIS bolgede mi sabit (robust)? Trend-following parametreleri
standart olmali, data'ya fit edilmemeli.

1. Parametre grid: ma_fast x ma_slow x breakout x vol_lb -> Sharpe dagilimi
2. Out-of-sample: 2016-2020 (IS) vs 2021-2026 (OOS) ayri
3. Leave-one-asset-out: tek varlik mi tasiyor?
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from itertools import product
from trend.data import fetch_prices
from trend.strategy import run_trend_backtest, TrendConfig, DEFAULT_UNIVERSE

def main():
    close = fetch_prices(DEFAULT_UNIVERSE, period="10y")
    print(f"{close.shape[1]} varlik, {len(close)} gun\n")

    # 1. PARAMETRE GRID
    print("=== 1. PARAMETRE ROBUSTLUGU (vol-target=15%) ===")
    fasts = [20, 50, 80]
    slows = [100, 150, 200]
    breakouts = [30, 50, 80]
    sharpes = []
    for f, s, bo in product(fasts, slows, breakouts):
        if f >= s:
            continue
        cfg = TrendConfig(ma_fast=f, ma_slow=s, breakout_lookback=bo, vol_target_annual=0.15)
        m = run_trend_backtest(close, cfg).metrics
        sharpes.append(m["sharpe"])
    arr = np.array(sharpes)
    print(f"  {len(arr)} kombo test edildi")
    print(f"  Sharpe: ort={arr.mean():.2f}  min={arr.min():.2f}  max={arr.max():.2f}  medyan={np.median(arr):.2f}")
    print(f"  Pozitif kombo orani: {(arr>0).mean()*100:.0f}%  (>0.7: {(arr>0.7).mean()*100:.0f}%)")
    print(f"  -> {'ROBUST (genis bolgede pozitif)' if (arr>0.5).mean()>0.8 else 'KIRILGAN (dar bolge)'}")

    # 2. OUT-OF-SAMPLE
    print("\n=== 2. OUT-OF-SAMPLE (parametreler sabit, donem ayri) ===")
    cfg = TrendConfig(vol_target_annual=0.15)
    full = run_trend_backtest(close, cfg)
    r = full.daily_returns
    for lo, hi, lbl in [(2016, 2020, "IS  2016-2020"), (2021, 2026, "OOS 2021-2026")]:
        sub = r[(r.index.year >= lo) & (r.index.year <= hi)]
        eq = (1+sub).cumprod(); dd = ((eq/eq.cummax())-1).min()
        sh = (sub.mean()-0.03/252)/sub.std()*np.sqrt(252)
        cagr = (1+sub).prod()**(252/len(sub))-1
        yearly = (1+sub).groupby(pd.Grouper(freq="YE")).prod()-1
        print(f"  {lbl}: CAGR={cagr*100:+.1f}%  Sharpe={sh:+.2f}  DD={dd*100:.0f}%  Yil+={(yearly>0).mean()*100:.0f}%")

    # 3. LEAVE-ONE-ASSET-OUT (robustluk: tek varlik mi tasiyor?)
    print("\n=== 3. LEAVE-ONE-ASSET-OUT (tek varlik bagimliligi) ===")
    base = run_trend_backtest(close, cfg).metrics["sharpe"]
    drops = []
    for col in close.columns:
        sub = close.drop(columns=[col])
        m = run_trend_backtest(sub, cfg).metrics
        drops.append((col, m["sharpe"]))
    drops.sort(key=lambda x: x[1])
    print(f"  Tam universe Sharpe: {base:.2f}")
    print(f"  En cok dusuren 3 varlik (cikinca):")
    for col, sh in drops[:3]:
        print(f"    -{col}: Sharpe {sh:.2f} (delta {sh-base:+.2f})")
    print(f"  -> {'ROBUST (tek varliga bagimli degil)' if drops[0][1] > base-0.2 else 'TEK VARLIK TASIYOR'}")

if __name__ == "__main__":
    main()
