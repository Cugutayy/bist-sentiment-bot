"""KOMBINE sistem: kripto EMA-3/15 sleeve + multi-asset trend blend.

Iki kazanan strateji birlestir, diversifikasyon faydasi:
- Kripto sleeve: EMA 3/15 long-flat (Sharpe 0.96, yuksek getiri)
- Multi-asset sleeve: 3-sinyal blend trend (Sharpe 0.91, stabil, uncorrelated)

Farkli agirliklar test + korelasyon analizi.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices
from trend.strategy import run_trend_backtest, TrendConfig, DEFAULT_UNIVERSE

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]

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

def stat(r, name, ann=365, rf=0.04):
    r=r.dropna()
    if len(r)<60: print(f"{name}: yetersiz"); return r
    cagr=(1+r).prod()**(ann/len(r))-1
    sh=(r.mean()-rf/ann)/r.std()*np.sqrt(ann)
    eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
    pm=((1+r).groupby(pd.Grouper(freq="ME")).prod()-1>0).mean()
    py=((1+r).groupby(pd.Grouper(freq="YE")).prod()-1>0).mean()
    print(f"{name:30s} CAGR={cagr*100:+6.1f}%  Sh={sh:+.2f}  DD={dd*100:+5.0f}%  Ay+={pm*100:.0f}%  Yil+={py*100:.0f}%")
    return r

if __name__=="__main__":
    cc = fetch_prices(CRYPTO, period="5y")
    mm = fetch_prices(DEFAULT_UNIVERSE, period="10y")
    print(f"Kripto {cc.shape[1]} coin, Multi {mm.shape[1]} varlik\n")

    crypto_r = crypto_ema_sleeve(cc)  # gunluk (365)
    multi_res = run_trend_backtest(mm, TrendConfig(vol_target_annual=0.20))
    multi_r = multi_res.daily_returns  # is-gunu (252)

    print("=== TEKIL SLEEVE'LER ===")
    stat(crypto_r, "Kripto EMA 3/15", ann=365)
    stat(multi_r, "Multi-asset blend", ann=252)

    # KOMBINE: ortak tarihlerde birlestir. Kripto 7/24, multi is-gunu.
    # Multi'yi kripto takvimine reindex (ffill yok, sadece ortak gunler)
    comb_idx = crypto_r.index.intersection(multi_r.index)
    cr = crypto_r.reindex(comb_idx); mr = multi_r.reindex(comb_idx)
    print(f"\nOrtak gun: {len(comb_idx)}  Kripto<->Multi korelasyon: {cr.corr(mr):+.2f}")

    print("\n=== KOMBINE (agirlik sweep) ===")
    for wc in [0.3, 0.5, 0.7]:
        comb = wc*cr + (1-wc)*mr
        stat(comb, f"Kombine kripto%{int(wc*100)}/multi%{int((1-wc)*100)}", ann=252)
    # son 3 yil
    print("\n=== Son 3 yil (kombine 50/50) ===")
    c50 = 0.5*cr+0.5*mr
    c50_3y = c50[c50.index>=c50.index[-1]-pd.Timedelta(days=365*3)]
    stat(c50_3y, "Kombine 50/50 son3y", ann=252)
