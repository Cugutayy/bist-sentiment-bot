"""Multi-asset trend-following (managed futures tarzi) — TUTARLILIK testi.

Tez: cok sayida KORELASYONSUZ piyasaya yayilmis trend-following =
puruzsuz equity curve. AQR/Winton/MAN AHL'nin onlarca yillik ~0.7-1.0
Sharpe, dusuk korelasyon, krizlerde bile pozitif sonuclari buradan gelir.

Universe (yfinance ETF/proxy):
  Hisse: SPY QQQ EFA EEM
  Tahvil: TLT IEF
  Emtia: GLD SLV USO DBC
  FX/Dolar: UUP
  Kripto: BTC-USD
  REIT: VNQ
Her biri trend-follow + vol-target (risk parity) + esit risk dagit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ["SPY","QQQ","EFA","EEM","TLT","IEF","GLD","SLV","USO","DBC","UUP","VNQ","BTC-USD"]
RF = 0.03/252
ANN = 252
COST = 0.0005  # 5bp ETF

def dl():
    df = yf.download(ASSETS, period="10y", interval="1d", auto_adjust=True, progress=False)
    close = df["Close"]
    close = close.dropna(axis=1, thresh=int(len(close)*0.4))
    print(f"Assets: {list(close.columns)}")
    return close

def stats(r, name, bench=None):
    r = r.dropna()
    if len(r)<60: print(f"{name}: yetersiz"); return r
    cagr=(1+r).prod()**(ANN/len(r))-1
    sh=(r.mean()-RF)/r.std()*np.sqrt(ANN)
    eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
    monthly=(1+r).groupby(pd.Grouper(freq="ME")).prod()-1
    pos_m=(monthly>0).mean()
    yearly=(1+r).groupby(pd.Grouper(freq="YE")).prod()-1
    pos_y=(yearly>0).mean()
    corr=r.corr(bench.reindex(r.index)) if bench is not None else 0
    print(f"{name:30s} CAGR={cagr*100:+6.1f}%  Sharpe={sh:+.2f}  MaxDD={dd*100:+5.0f}%  Poz.Ay={pos_m*100:.0f}%  Poz.Yil={pos_y*100:.0f}%  korSPY={corr:+.2f}")
    return r

def trend(close, ma=100, vol_target=0.15, vol_lb=50, longshort=True):
    """Risk-parity trend + PORTFOY-seviyesi vol-target (ex-post scaling)."""
    rets=close.pct_change()
    sig=(close>close.rolling(ma).mean()).astype(float)
    if longshort: sig=sig*2-1
    sig=sig.shift(1)
    cvol=rets.rolling(vol_lb).std()*np.sqrt(ANN)
    # risk parity: her asset 1/vol agirlik (esit risk katkisi)
    inv_vol=(1.0/cvol).clip(0,50).shift(1)
    pos=sig*inv_vol
    # normalize: gross exposure = 1 (sum |w| = 1)
    gross_exp=pos.abs().sum(axis=1).replace(0,np.nan)
    pos=pos.div(gross_exp,axis=0).fillna(0)
    raw=(pos*rets).sum(axis=1)
    turn=pos.diff().abs().sum(axis=1)
    raw_net=(raw-turn*COST).dropna()
    # PORTFOY-seviyesi vol target: realized 60g vol -> hedef
    realized=raw_net.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    scale=(vol_target/realized).shift(1).clip(0,5).fillna(1.0)
    return (raw_net*scale).dropna()

if __name__=="__main__":
    close=dl()
    spy=close["SPY"].pct_change()
    print(f"\nDonem: {close.index[0].date()} -> {close.index[-1].date()}\n")
    print("=== BENCHMARK ===")
    stats(spy,"SPY buy-hold",spy)
    print("\n=== MULTI-ASSET TREND (long-short, risk-parity) ===")
    for ma in [50,100,150,200]:
        stats(trend(close,ma=ma,longshort=True),f"MA{ma} LS",spy)
    print("\n=== VOL-TARGET SWEEP (MA100 LS) ===")
    for vt in [0.10,0.15,0.20,0.30]:
        stats(trend(close,ma=100,vol_target=vt,longshort=True),f"MA100 volT={vt:.0%}",spy)
    print("\n=== Son 5 yil (recent) MA100 LS volT=15% ===")
    r=trend(close,ma=100,vol_target=0.15,longshort=True)
    r5=r[r.index>=r.index[-1]-pd.Timedelta(days=365*5)]
    stats(r5,"Son 5y",spy)
