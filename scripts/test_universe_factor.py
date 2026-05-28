"""BIST 100 genis universe'de faktor alpha testi — RIGOROUS.

Amac: BIST 30 yerine BIST 100 kullanmak faktor secim alpha'sini
artiriyor mu? Daha cok isim = daha cok cross-sectional dispersion.

Sentiment'e bagimli DEGIL (saf fiyat faktoru), o yuzden BIST50'deki
news-NaN sorunu gecerli degil.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

# BIST 100 — likit isimler (BIST 30 + ~70 ek). Bazilari delist/yfinance fail olabilir.
BIST100 = [
    # BIST 30
    "THYAO","ASELS","GARAN","AKBNK","YKBNK","ISCTR","HALKB","VAKBN","KCHOL","SAHOL",
    "TUPRS","SISE","EREGL","BIMAS","FROTO","TOASO","TCELL","TAVHL","PGSUS","EKGYO",
    "PETKM","TTKOM","ARCLK","ENKAI","TKFEN","SASA","GUBRF","DOHOL","MGROS","SOKM",
    # 31-100 (likit orta/buyuk cap)
    "ALARK","AGHOL","AKSEN","AKSA","ANSGR","ASUZU","AYDEM","AYGAZ","BAGFS","BERA",
    "BIENY","BRSAN","BRYAT","BUCIM","CCOLA","CIMSA","CWENE","DOAS","ECILC","ECZYT",
    "EGEEN","ENERY","ENJSA","EUPWR","FENER","GESAN","GLYHO","GOLTS","GSDHO","HEKTS",
    "IPEKE","ISMEN","IZMDC","KARSN","KARTN","KAYSE","KCAER","KLSER","KMPUR","KONTR",
    "KORDS","KOZAA","KOZAL","KRDMD","MAVI","MIATK","MPARK","ODAS","OTKAR","OYAKC",
    "PENTA","PETUN","POLHO","QUAGR","SELEC","SKBNK","SMRTG","SNGYO","TKNSA","TMSN",
    "TSKB","TTRAK","TUKAS","TURSG","ULKER","VESBE","VESTL","YEOTK","ZOREN","AEFES",
]

def download(tickers, years=5):
    syms = [t+".IS" for t in tickers] + ["XU100.IS"]
    print(f"Downloading {len(syms)} symbols...")
    df = yf.download(syms, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    close = df["Close"]
    # Drop tickers with too much missing
    valid = close.columns[close.notna().mean() > 0.8]
    close = close[valid]
    print(f"  Valid: {len(valid)} symbols (>{0.8:.0%} data)")
    return close

def factor_scores(close):
    """Cross-sectional factor composite per date."""
    rets = close.pct_change()
    # Market = equal-weight ex-XU100
    stock_cols = [c for c in close.columns if c != "XU100.IS"]
    mkt = rets[stock_cols].mean(axis=1)
    panel = {}
    for c in stock_cols:
        r = rets[c]
        resid = r - mkt
        idio60 = resid.rolling(60, min_periods=20).std()
        vol20 = r.rolling(20).std()
        mom20 = close[c].pct_change(20)
        mom60 = close[c].pct_change(60)
        panel[c] = pd.DataFrame({"idio60":idio60,"vol20":vol20,"mom20":mom20,"mom60":mom60,"fwd1":r.shift(-1)})
    return panel, stock_cols

def csz(mat):
    return mat.sub(mat.mean(axis=1),axis=0).div(mat.std(axis=1).replace(0,np.nan),axis=0)

def test(close, universe_name, k=5, rebal=10):
    panel, cols = factor_scores(close)
    dates = close.index
    # Build factor matrices [date x ticker]
    def mat(field):
        return pd.DataFrame({c:panel[c][field] for c in cols})
    idio=mat("idio60"); vol=mat("vol20"); m20=mat("mom20"); m60=mat("mom60"); fwd=mat("fwd1")
    comp = -csz(idio) - 0.5*csz(vol) + 0.5*csz(m20) + 0.3*csz(m60)
    # equal-weight universe benchmark
    uni_ret = fwd.mean(axis=1)
    # XU100
    xu_ret = close["XU100.IS"].pct_change().shift(-1)
    RF=0.40/252
    ONE_WAY=0.0035
    rebal_idx=set(range(0,len(dates),rebal))
    cur=set(); prev=set(); out=[]
    for i,dt in enumerate(dates):
        row=comp.loc[dt].dropna()
        if i in rebal_idx and len(row)>=k:
            cur=set(row.sort_values(ascending=False).head(k).index)
        if not cur: out.append(np.nan); continue
        fr=fwd.loc[dt, list(cur)].mean()
        cost=0.0
        if i in rebal_idx:
            t=len(cur.symmetric_difference(prev)); cost=t/k*ONE_WAY; prev=cur.copy()
        out.append(fr-cost)
    sp=pd.Series(out,index=dates).dropna()
    def st(r,rf=True):
        r=r.dropna(); cagr=(1+r).prod()**(252/len(r))-1
        sh=(r.mean()-(RF if rf else 0))/r.std()*np.sqrt(252)
        eq=(1+r).cumprod(); dd=((eq/eq.cummax())-1).min()
        return cagr,sh,dd
    c1,s1,d1=st(sp)
    cu,su,du=st(uni_ret.reindex(sp.index))
    cx,sx,dx=st(xu_ret.reindex(sp.index))
    print(f'\\n=== {universe_name} (n={len(cols)}, k={k}, rebal={rebal}g) ===')
    print(f'  Faktor top-{k}: CAGR={c1*100:+6.1f}%  Sharpe(rf)={s1:+.2f}  MaxDD={d1*100:.0f}%')
    print(f'  Universe EW   : CAGR={cu*100:+6.1f}%  Sharpe(rf)={su:+.2f}  MaxDD={du*100:.0f}%')
    print(f'  XU100         : CAGR={cx*100:+6.1f}%  Sharpe(rf)={sx:+.2f}  MaxDD={dx*100:.0f}%')
    print(f'  Faktor EXCESS vs universe: {(c1-cu)*100:+.1f}pp/yil')

if __name__=="__main__":
    close=download(BIST100)
    # BIST30 subset
    b30=[t+".IS" for t in BIST100[:30] if t+".IS" in close.columns]+["XU100.IS"]
    test(close[b30],"BIST 30", k=3, rebal=10)
    test(close,"BIST 100", k=3, rebal=10)
    test(close,"BIST 100", k=5, rebal=10)
    test(close,"BIST 100", k=10, rebal=10)
    test(close,"BIST 100", k=5, rebal=20)
