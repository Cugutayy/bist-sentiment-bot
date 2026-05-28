"""Arastirma-tabanli iyilestirmeler: EMA + ADX filtresi.

Web arastirmasi (arxiv 2511.00665, Cinli quant platformlari) bulgulari:
1. EMA > SMA (daha hizli tepki, daha iyi giris)
2. ADX trend-gucu filtresi: sadece ADX>esik iken MA sinyali al
   -> choppy/trendsiz donemde whipsaw keser (2021-23 'drought' defansi)
3. MACD+ADX combo akademik testte EMA crossover'i gecti

Test: kripto + multi-asset uzerinde EMA vs SMA, ADX filtreli vs filtresiz.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]
MULTI = ["SPY","QQQ","IWM","EFA","EEM","XLE","XLF","XLK","XLV","XLI","TLT","IEF","SHY","LQD","HYG",
         "GLD","SLV","USO","UNG","DBC","DBA","CPER","UUP","FXE","FXY","VNQ","BTC-USD","ETH-USD"]

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def adx(high, low, close, n=14):
    """ADX hesabi (close-only proxy: high=low=close yoksa). Trend gucu 0-100."""
    # close-only: TR ~ |close.diff|, DM yaklasimi
    up = close.diff().clip(lower=0)
    dn = (-close.diff()).clip(lower=0)
    atr = close.diff().abs().rolling(n).mean()
    pdi = 100 * up.rolling(n).mean() / atr.replace(0, np.nan)
    ndi = 100 * dn.rolling(n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(n).mean()

def backtest(close, fast, slow, use_ema=True, adx_filter=None, vol_target=0.30, ann=365, cost=0.0010, longshort=False):
    rets = close.pct_change()
    if use_ema:
        sig_raw = np.sign(close.apply(lambda c: ema(c, fast)) - close.apply(lambda c: ema(c, slow)))
    else:
        sig_raw = np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())
    if not longshort:
        sig_raw = sig_raw.clip(lower=0)
    # ADX filtresi: trend zayifsa pozisyon alma (flat)
    if adx_filter is not None:
        adx_vals = close.apply(lambda c: adx(c, c, c))
        trend_ok = (adx_vals > adx_filter).astype(float)
        sig_raw = sig_raw * trend_ok
    sig = sig_raw.shift(1)
    cvol = rets.rolling(30).std()*np.sqrt(ann)
    inv = (vol_target/cvol).clip(0,3).shift(1)
    pos = sig*inv/close.shape[1]
    raw = (pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost
    raw = raw.dropna()
    # portfoy vol-target
    realized = raw.rolling(60,min_periods=20).std()*np.sqrt(ann)
    scale = (vol_target/realized).shift(1).clip(0,5).fillna(1.0)
    return (raw*scale).dropna()

def stat(r, name, ann=365, rf=0.05):
    r=r.dropna()
    if len(r)<60: print(f"{name}: yetersiz"); return
    cagr=(1+r).prod()**(ann/len(r))-1
    sh=(r.mean()-rf/ann)/r.std()*np.sqrt(ann)
    eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
    pm=((1+r).groupby(pd.Grouper(freq="ME")).prod()-1>0).mean()
    print(f"{name:32s} CAGR={cagr*100:+6.1f}%  Sh={sh:+.2f}  DD={dd*100:+5.0f}%  Ay+={pm*100:.0f}%")

if __name__=="__main__":
    print("=== KRIPTO: EMA vs SMA, ADX filtresi ===")
    c = fetch_prices(CRYPTO, period="5y")
    print(f"{c.shape[1]} coin\n")
    stat(backtest(c, 3,15, use_ema=False, adx_filter=None), "SMA 3/15 (baseline)")
    stat(backtest(c, 3,15, use_ema=True,  adx_filter=None), "EMA 3/15")
    stat(backtest(c, 10,50, use_ema=True, adx_filter=None), "EMA 10/50")
    stat(backtest(c, 12,50, use_ema=True, adx_filter=None), "EMA 12/50 (altFINS)")
    stat(backtest(c, 9,21, use_ema=True,  adx_filter=None), "EMA 9/21 (BTC profit-factor)")
    print("  -- ADX filtreli (sadece trend gucluyken) --")
    stat(backtest(c, 9,21, use_ema=True, adx_filter=20), "EMA 9/21 + ADX>20")
    stat(backtest(c, 9,21, use_ema=True, adx_filter=25), "EMA 9/21 + ADX>25")
    stat(backtest(c, 12,50, use_ema=True, adx_filter=25),"EMA 12/50 + ADX>25")
    stat(backtest(c, 10,50, use_ema=True, adx_filter=25, longshort=True),"EMA 10/50 + ADX>25 L/S")

    print("\n=== MULTI-ASSET: EMA + ADX ===")
    m = fetch_prices(MULTI, period="10y")
    print(f"{m.shape[1]} varlik\n")
    stat(backtest(m, 50,200, use_ema=False, adx_filter=None, vol_target=0.15, ann=252, cost=0.0005, longshort=True), "SMA 50/200 LS (baseline)", ann=252, rf=0.03)
    stat(backtest(m, 50,200, use_ema=True,  adx_filter=None, vol_target=0.15, ann=252, cost=0.0005, longshort=True), "EMA 50/200 LS", ann=252, rf=0.03)
    stat(backtest(m, 20,100, use_ema=True,  adx_filter=None, vol_target=0.15, ann=252, cost=0.0005, longshort=True), "EMA 20/100 LS", ann=252, rf=0.03)
    stat(backtest(m, 20,100, use_ema=True,  adx_filter=20, vol_target=0.15, ann=252, cost=0.0005, longshort=True), "EMA 20/100 + ADX>20 LS", ann=252, rf=0.03)
