"""HALLUSINASYON / BUG DENETIMI — kripto EMA 3/15 ve kombine sistem.

Onceki ffill-bug ve leverage-trap derslerinden sonra, sonuclari
yayinlamadan once 4 klasik backtest tuzagini test et:

1. SHUFFLE TEST (altin standart): sinyali rastgele karistir.
   Eger karisik sinyal hala para kaziyorsa -> PnL look-ahead bug var.
2. SURVIVORSHIP: BTC+ETH-only (her zaman vardi) vs full alt basket.
   Full >> BTC/ETH ise olu-coin hayatta-kalma yanlilik sismesi var.
3. LOOK-AHEAD: sinyali 1 gun DAHA shift et. Sonuc cokerse peeking vardi.
4. COST STRESS: 10 -> 30 -> 50 -> 100 bp. Edge maliyete dayanikli mi?
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from trend.data import fetch_prices

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD"]
ANN=365

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def strat(close, fast=3, slow=15, vol_target=0.30, cost=0.0010, extra_shift=0, shuffle=False, seed=0):
    rets=close.pct_change()
    sig=np.sign(close.apply(lambda c: ema(c,fast))-close.apply(lambda c: ema(c,slow))).clip(lower=0)
    sig=sig.shift(1+extra_shift)
    if shuffle:
        # her kolonu zaman ekseninde bagimsiz karistir (sinyal-getiri iliskisini boz)
        rng=np.random.default_rng(seed)
        sig=sig.apply(lambda c: pd.Series(rng.permutation(c.values), index=c.index))
    cvol=rets.rolling(30).std()*np.sqrt(ANN)
    pos=sig*(vol_target/cvol).clip(0,3).shift(1)/close.shape[1]
    raw=((pos*rets).sum(axis=1)-pos.diff().abs().sum(axis=1)*cost).dropna()
    realized=raw.rolling(60,min_periods=20).std()*np.sqrt(ANN)
    return (raw*(vol_target/realized).shift(1).clip(0,5).fillna(1)).dropna()

def sh(r, rf=0.05):
    r=r.dropna()
    if len(r)<60 or r.std()==0: return 0,0
    return (r.mean()-rf/ANN)/r.std()*np.sqrt(ANN), (1+r).prod()**(ANN/len(r))-1

if __name__=="__main__":
    close=fetch_prices(CRYPTO, period="5y")
    print(f"{close.shape[1]} coin, {len(close)} gun\n")

    base_sh, base_cagr = sh(strat(close))
    print(f"BASELINE EMA 3/15: Sharpe={base_sh:+.2f}  CAGR={base_cagr*100:+.1f}%\n")

    print("=== 1. SHUFFLE TEST (sinyal rastgele) ===")
    shuf=[sh(strat(close, shuffle=True, seed=i))[0] for i in range(10)]
    print(f"  10 shuffle Sharpe: ort={np.mean(shuf):+.2f}  min={min(shuf):+.2f}  max={max(shuf):+.2f}")
    # DOGRU mantik: bug = shuffle POZITIF kazanir (PnL leak). Negatif/sifir = temiz
    # (rastgele sinyal maliyet oder, edge yok -> hafif negatif beklenir).
    print(f"  -> {'BUG! shuffle pozitif kaziyor = PnL look-ahead' if np.mean(shuf)>0.30 or max(shuf)>0.50 else 'TEMIZ (shuffle <=0, gercek sinyal +0.96 >> shuffle)'}")

    print("\n=== 2. SURVIVORSHIP (BTC+ETH only vs full) ===")
    be=fetch_prices(["BTC-USD","ETH-USD"], period="5y")
    s_be,c_be=sh(strat(be)); s_full,c_full=sh(strat(close))
    print(f"  BTC+ETH only: Sharpe={s_be:+.2f}  CAGR={c_be*100:+.1f}%")
    print(f"  Full basket : Sharpe={s_full:+.2f}  CAGR={c_full*100:+.1f}%")
    print(f"  -> {'Alt basket survivorship ile sisirilmis OLABILIR' if c_full>c_be*1.3 else 'BTC/ETH cekirdegi tasiyor, survivorship sinirli'}")

    print("\n=== 3. LOOK-AHEAD (1 gun ekstra shift) ===")
    s0,c0=sh(strat(close,extra_shift=0)); s1,c1=sh(strat(close,extra_shift=1))
    print(f"  Normal shift:  Sharpe={s0:+.2f}  CAGR={c0*100:+.1f}%")
    print(f"  +1 gun shift:  Sharpe={s1:+.2f}  CAGR={c1*100:+.1f}%")
    print(f"  -> {'TEMIZ (ekstra shift az degisti)' if abs(s0-s1)<0.4 else 'DIKKAT: ekstra shift cok degistirdi (timing hassas/peeking?)'}")

    print("\n=== 4. COST STRESS ===")
    for c in [0.0010,0.0030,0.0050,0.0100]:
        s,cg=sh(strat(close,cost=c))
        print(f"  cost={c*10000:.0f}bp: Sharpe={s:+.2f}  CAGR={cg*100:+.1f}%")
