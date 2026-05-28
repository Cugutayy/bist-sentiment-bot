"""Multi-asset trend v2 — recent donemi guclendirme.

Iyilestirmeler:
1. Daha cok varlik (sektor ETF + daha cok emtia/FX/bond) = daha cok diversifikasyon
2. Coklu trend sinyali blend: MA crossover + Donchian breakout + TS-momentum (3 horizon)
   -> tek MA whipsaw'ina bagimliligi azaltir (drought defansi)
3. Per-asset vol-target + portfoy-seviyesi vol-target
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

# Genis universe — daha cok korelasyonsuz piyasa
ASSETS = [
    # Hisse (bolgesel + sektor)
    "SPY","QQQ","IWM","EFA","EEM","XLE","XLF","XLK","XLV","XLI",
    # Tahvil
    "TLT","IEF","SHY","LQD","HYG",
    # Emtia
    "GLD","SLV","USO","UNG","DBC","DBA","CPER",
    # FX / Dolar
    "UUP","FXE","FXY",
    # REIT + kripto
    "VNQ","BTC-USD","ETH-USD",
]
RF = 0.03/252; ANN = 252; COST = 0.0005

def dl():
    df = yf.download(ASSETS, period="10y", interval="1d", auto_adjust=True, progress=False)
    close = df["Close"].dropna(axis=1, thresh=int(len(df)*0.4))
    print(f"{len(close.columns)} varlik yuklendi")
    return close

def stats(r, name, bench=None):
    r=r.dropna()
    if len(r)<60: print(f"{name}: yetersiz"); return r
    cagr=(1+r).prod()**(ANN/len(r))-1
    sh=(r.mean()-RF)/r.std()*np.sqrt(ANN)
    eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
    pm=((1+r).groupby(pd.Grouper(freq="ME")).prod()-1>0).mean()
    py=((1+r).groupby(pd.Grouper(freq="YE")).prod()-1>0).mean()
    corr=r.corr(bench.reindex(r.index)) if bench is not None else 0
    print(f"{name:26s} CAGR={cagr*100:+6.1f}%  Sh={sh:+.2f}  DD={dd*100:+5.0f}%  Ay+={pm*100:.0f}%  Yil+={py*100:.0f}%  kor={corr:+.2f}")
    return r

def blended_signal(close):
    """3 sinyal ortalamasi: MA crossover, Donchian breakout, TS-momentum."""
    # 1. MA crossover (50 vs 200)
    s_ma = np.sign(close.rolling(50).mean() - close.rolling(200).mean())
    # 2. Donchian breakout (50g): fiyat 50g max'a yakinsa +1, min'e yakinsa -1
    hi = close.rolling(50).max(); lo = close.rolling(50).min()
    pos_in_range = (close - lo)/(hi - lo).replace(0,np.nan)
    s_break = np.where(pos_in_range>0.5, 1, -1)
    s_break = pd.DataFrame(s_break, index=close.index, columns=close.columns)
    # 3. TS-momentum (3 horizon: 60/120/250g getiri isareti)
    s_mom = (np.sign(close.pct_change(60)) + np.sign(close.pct_change(120)) + np.sign(close.pct_change(250)))/3
    blend = (s_ma + s_break + s_mom)/3.0  # -1..+1
    return blend.shift(1)

def run(close, vol_target=0.15, vol_lb=50):
    rets=close.pct_change()
    sig=blended_signal(close)
    cvol=rets.rolling(vol_lb).std()*np.sqrt(ANN)
    inv=(1/cvol).clip(0,50).shift(1)
    pos=sig*inv
    pos=pos.div(pos.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    raw=(pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*COST
    raw=raw.dropna()
    realized=raw.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

if __name__=="__main__":
    close=dl()
    spy=close["SPY"].pct_change()
    print(f"\nDonem: {close.index[0].date()} -> {close.index[-1].date()}\n")
    stats(spy,"SPY buy-hold",spy)
    print("--- Blended trend (genis universe) ---")
    full=run(close,vol_target=0.15)
    stats(full,"Blended volT=15% (10y)",spy)
    for vt in [0.10,0.20,0.25]:
        stats(run(close,vol_target=vt),f"Blended volT={vt:.0%}",spy)
    print("--- Donem analizi (volT=15%) ---")
    for lo,hi in [(2016,2020),(2021,2026)]:
        sub=full[(full.index.year>=lo)&(full.index.year<=hi)]
        stats(sub,f"{lo}-{hi}",spy)
    print("--- Yil yil (volT=20%) ---")
    r20=run(close,vol_target=0.20)
    for y in range(2016,2027):
        rt=r20[r20.index.year==y]; st=spy[spy.index.year==y].reindex(rt.index).dropna()
        if len(rt)<10: continue
        print(f"  {y}: trend {((1+rt).prod()-1)*100:+6.1f}%   SPY {((1+st).prod()-1)*100:+6.1f}%")
