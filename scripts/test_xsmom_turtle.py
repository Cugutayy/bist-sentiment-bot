"""Iki arastirma-tabanli strateji, SKEPTIK test:

A. CROSS-SECTIONAL MOMENTUM (akademik: 28g lookback/5g hold -> Sharpe 1.51)
   - Her rebalance: coinleri gecmis getiriye gore sirala, en iyileri long
   - Time-series MA'dan FARKLI: goreceli secim (en guclu coinler)
   - Skeptik: makale maliyet-oncesi + 2013-2023 (cilgin erken yillar).
     Gercekci 30bp + recent donem (2021-26) ile test.

B. TURTLE (海龟): 20g yuksek kir -> al, 10g dusuk -> sat (Donchian asimetrik)

Audit: shuffle + cost-stress + recent-only.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices

CRYPTO=["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]
ANN=365

def stat(r, name, rf=0.05):
    r=r.dropna()
    if len(r)<60 or r.std()==0: print(f"{name}: yetersiz"); return r
    cagr=(1+r).prod()**(ANN/len(r))-1
    sh=(r.mean()-rf/ANN)/r.std()*np.sqrt(ANN)
    eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
    pm=((1+r).groupby(pd.Grouper(freq="ME")).prod()-1>0).mean()
    print(f"{name:34s} CAGR={cagr*100:+6.1f}%  Sh={sh:+.2f}  DD={dd*100:+5.0f}%  Ay+={pm*100:.0f}%")
    return r

def xs_momentum(close, lookback=28, hold=5, top_k=3, cost=0.0030, longshort=False, vol_target=0.40):
    """Cross-sectional: gecmis getiri sirala, top_k long (+ bottom_k short)."""
    rets=close.pct_change()
    mom=close.pct_change(lookback)  # gecmis getiri (momentum skoru)
    dates=close.index
    n=close.shape[1]
    pos=pd.DataFrame(0.0, index=dates, columns=close.columns)
    rebal=range(0,len(dates),hold)
    cur_l, cur_s=[], []
    for i,dt in enumerate(dates):
        if i in rebal:
            row=mom.loc[dt].dropna()
            if len(row)>=top_k*(2 if longshort else 1):
                srt=row.sort_values(ascending=False)
                cur_l=list(srt.head(top_k).index)
                cur_s=list(srt.tail(top_k).index) if longshort else []
        # vol-scaled equal weight
        cvol=rets[cur_l].iloc[max(0,i-30):i].std()*np.sqrt(ANN) if cur_l else None
        for t in cur_l: pos.loc[dt,t]=1.0/top_k
        for t in cur_s: pos.loc[dt,t]=-1.0/top_k
    pos=pos.shift(1)
    raw=((pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost).dropna()
    realized=raw.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

def turtle(close, entry=20, exit=10, cost=0.0030, vol_target=0.40):
    """20g yuksek kir -> long, 10g dusuk -> cik (long-flat)."""
    rets=close.pct_change()
    hi=close.rolling(entry).max(); lo=close.rolling(exit).min()
    sig=pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    sig=sig.where(~(close>=hi.shift(1)), 1.0)   # breakout up -> long
    sig=sig.where(~(close<=lo.shift(1)), 0.0)   # breakdown -> flat
    sig=sig.ffill().fillna(0.0).shift(1)
    cvol=rets.rolling(30).std()*np.sqrt(ANN)
    pos=sig*(vol_target/cvol).clip(0,3).shift(1)/close.shape[1]
    raw=((pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost).dropna()
    realized=raw.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

if __name__=="__main__":
    close=fetch_prices(CRYPTO, period="5y")
    print(f"{close.shape[1]} coin, {len(close)} gun\n")
    print("=== A. CROSS-SECTIONAL MOMENTUM (gercekci 30bp) ===")
    for lb,hd in [(28,5),(28,7),(14,5),(60,10),(90,10)]:
        stat(xs_momentum(close,lb,hd,top_k=3,cost=0.0030), f"XS-mom {lb}g/{hd}g hold long-only")
    stat(xs_momentum(close,28,5,top_k=3,cost=0.0030,longshort=True), "XS-mom 28/5 LONG-SHORT")
    print("\n=== B. TURTLE 20/10 (gercekci 30bp) ===")
    for e,x in [(20,10),(20,20),(55,20),(10,5)]:
        stat(turtle(close,e,x,cost=0.0030), f"Turtle {e}/{x}")
    print("\n=== En iyi XS-mom: cost stress + recent ===")
    for c in [0.0010,0.0030,0.0050]:
        stat(xs_momentum(close,28,5,top_k=3,cost=c), f"XS-mom 28/5 cost={c*10000:.0f}bp")
    best=xs_momentum(close,28,5,top_k=3,cost=0.0030)
    stat(best[best.index.year>=2023], "XS-mom 28/5 son ~3yil (2023+)")
