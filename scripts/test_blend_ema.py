"""Multi-asset blend: SMA vs EMA MA-crossover bileseni.

EMA kripto 3/15'te SMA'yi gecti (Sharpe 0.75->0.99). Multi-asset
blend'in MA-crossover bileseni de SMA->EMA olunca iyilesiyor mu?
Diger 2 sinyal (breakout, TS-momentum) ayni kalir.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices
from trend.strategy import (DEFAULT_UNIVERSE, signal_breakout, signal_ts_momentum,
                            compute_metrics, TrendConfig)
ANN=252

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def run(close, use_ema_cross, fast=50, slow=200, vol_target=0.15, cost=0.0005):
    rets=close.pct_change()
    if use_ema_cross:
        s_ma=np.sign(close.apply(lambda c: ema(c,fast))-close.apply(lambda c: ema(c,slow)))
    else:
        s_ma=np.sign(close.rolling(fast).mean()-close.rolling(slow).mean())
    s_bo=signal_breakout(close,50)
    s_mo=signal_ts_momentum(close,(60,120,250))
    sig=((s_ma+s_bo+s_mo)/3.0).shift(1)
    cvol=rets.rolling(50).std()*np.sqrt(ANN)
    pos=sig*(1/cvol).clip(0,50).shift(1)
    pos=pos.div(pos.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    raw=((pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost).dropna()
    realized=raw.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

def show(r,name):
    m=compute_metrics(r)
    print(f"{name:28s} CAGR={m['cagr']*100:+6.1f}%  Sh={m['sharpe']:+.2f}  Sortino={m['sortino']:+.2f}  DD={m['max_dd']*100:.0f}%  Yil+={m['positive_years_pct']*100:.0f}%")

if __name__=="__main__":
    close=fetch_prices(DEFAULT_UNIVERSE, period="10y")
    print(f"{close.shape[1]} varlik\n")
    print("=== Blend MA-crossover bileseni: SMA vs EMA ===")
    for f,s in [(50,200),(20,100),(30,150)]:
        show(run(close,False,f,s), f"SMA {f}/{s}")
        show(run(close,True, f,s), f"EMA {f}/{s}")
        print()
    # recent donem (EMA daha responsive -> drought'ta fark eder mi)
    print("=== Son 5 yil (2021-2026) ===")
    for f,s in [(50,200),(20,100)]:
        rs=run(close,False,f,s); re=run(close,True,f,s)
        rs5=rs[rs.index.year>=2021]; re5=re[re.index.year>=2021]
        show(rs5,f"SMA {f}/{s} son5y"); show(re5,f"EMA {f}/{s} son5y")
        print()
