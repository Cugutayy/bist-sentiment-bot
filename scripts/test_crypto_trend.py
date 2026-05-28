"""Kripto trend-following testi — RIGOROUS, tutarlilik odakli.

Strateji: time-series momentum (trend-following) + vol-targeting.
- Her coin icin: trend up (fiyat > N-gun MA) ise long, degilse flat (cash)
- Vol-targeting: pozisyon boyutu coin volatilitesi ile ters orantili (risk parity)
- Portfolio = coinler arasi esit risk

Olcum: CAGR, Sharpe, MaxDD, POZITIF AY ORANI (tutarlilik!), yillik breakdown.
USD-bazli (risk-free ~%5).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

CRYPTOS = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD",
           "DOGE-USD","AVAX-USD","LINK-USD","DOT-USD","MATIC-USD","LTC-USD"]
RF = 0.05/365  # USD risk-free gunluk (kripto 7/24, 365 gun)
ANN = 365
COST = 0.0010  # 10bp per trade (kripto spot ~%0.1)

def dl():
    df = yf.download(CRYPTOS, period="5y", interval="1d", auto_adjust=True, progress=False)
    close = df["Close"]
    close = close.dropna(axis=1, thresh=int(len(close)*0.5))
    print(f"Coins: {list(close.columns)}")
    return close

def stats(r, name, bench=None):
    r = r.dropna()
    if len(r) < 30: print(f"{name}: yetersiz veri"); return
    cagr = (1+r).prod()**(ANN/len(r)) - 1
    sh = (r.mean()-RF)/r.std()*np.sqrt(ANN)
    eq = (1+r).cumprod(); dd = ((eq/eq.cummax())-1).min()
    # pozitif ay orani (tutarlilik)
    monthly = (1+r).resample("ME").prod()-1 if hasattr(r.index,'freq') else (1+r).groupby(pd.Grouper(freq="ME")).prod()-1
    pos_m = (monthly>0).mean()
    corr = r.corr(bench.reindex(r.index)) if bench is not None else 0
    print(f"{name:34s} CAGR={cagr*100:+7.1f}%  Sharpe={sh:+.2f}  MaxDD={dd*100:+5.0f}%  Poz.Ay={pos_m*100:.0f}%  kor={corr:+.2f}")
    return r

def trend_strategy(close, ma_days=100, vol_target=0.40, vol_lb=30, longshort=False):
    rets = close.pct_change()
    # trend sinyali: fiyat > MA
    ma = close.rolling(ma_days).mean()
    signal = (close > ma).astype(float)  # 1 long, 0 flat
    if longshort:
        signal = signal*2 - 1  # +1 / -1
    signal = signal.shift(1)  # lag — dunku sinyal
    # vol-targeting per coin (risk parity)
    coin_vol = rets.rolling(vol_lb).std()*np.sqrt(ANN)
    coin_w = (vol_target/coin_vol).clip(0, 3).shift(1)  # her coin hedef vol'a scale
    pos = signal * coin_w
    # esit dagit (n coin)
    n = close.shape[1]
    pos = pos / n
    # portfoy getiri
    gross = (pos * rets).sum(axis=1)
    # cost
    turn = pos.diff().abs().sum(axis=1)
    net = gross - turn*COST
    return net.dropna(), pos

if __name__ == "__main__":
    close = dl()
    btc = close["BTC-USD"].pct_change()
    print(f"\nDonem: {close.index[0].date()} -> {close.index[-1].date()}\n")
    print("=== BENCHMARK ===")
    stats(btc, "BTC buy-hold", btc)
    ew = close.pct_change().mean(axis=1)
    stats(ew, "Equal-weight kripto buy-hold", btc)
    print("\n=== TREND-FOLLOWING (long-flat) ===")
    for ma in [50, 100, 150, 200]:
        net,_ = trend_strategy(close, ma_days=ma, longshort=False)
        stats(net, f"Trend long-flat MA{ma}", btc)
    print("\n=== TREND-FOLLOWING (long-short) ===")
    for ma in [50, 100, 200]:
        net,_ = trend_strategy(close, ma_days=ma, longshort=True)
        stats(net, f"Trend long-short MA{ma}", btc)
    print("\n=== VOL-TARGET SWEEP (long-flat MA100) ===")
    for vt in [0.30, 0.50, 0.70, 1.00]:
        net,_ = trend_strategy(close, ma_days=100, vol_target=vt, longshort=False)
        stats(net, f"Trend MA100 volT={vt:.0%}", btc)
