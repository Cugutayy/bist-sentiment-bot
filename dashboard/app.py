"""BIST Sentiment Bot — Streamlit Dashboard.

Kullanım yerel:
    streamlit run dashboard/app.py

Cloud deploy (Streamlit Cloud — ücretsiz):
    1. https://share.streamlit.io/ → Sign in with GitHub
    2. New app → bist-sentiment-bot reposu seç
    3. Main file: dashboard/app.py
    4. Deploy

NOT: Cloud'da data/ klasörü boş gelir (gitignore). Cron job bunu
sürekli güncellemediği için cloud dashboard sadece "ilk push'taki snapshot"u
gösterir. Sürekli güncel veri için cron'u local PC veya VPS'te tut, sadece
artifact'ları (parquet) repo'ya commit eden bir sub-script ekle (Faz 6).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Repo root'u import path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT, SETTINGS

st.set_page_config(
    page_title="BIST Sentiment Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("📊 BIST Sentiment Bot")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Sayfa",
    [
        "🏠 Özet",
        "📈 Backtest",
        "🎯 Bugünün Shortlist'i",
        "📝 Paper Trading",
        "💬 Sentiment",
        "🔍 Veri Kalitesi",
        "⚙️ Konfig & Mimari",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Honest System**\n\n"
    "Bu sistem yatırım tavsiyesi vermez. Backtest sonuçları "
    "survivorship bias içerir (×0.85 ile düzelt).\n\n"
    "Kod: [github.com/Cugutayy/bist-sentiment-bot](https://github.com/Cugutayy/bist-sentiment-bot)"
)


# ─── Helper'lar ─────────────────────────────────────────────────────
BT_DIR = ROOT / "data" / "processed" / "backtests"
PAPER_LOG = ROOT / "data" / "processed" / "paper_trades.parquet"
SHORTLIST_DIR = ROOT / "data" / "processed"


@st.cache_data
def load_latest_backtest():
    runs = sorted(BT_DIR.glob("*/"), reverse=True)
    if not runs:
        return None, None, None
    run_dir = runs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(run_dir / "daily.parquet").set_index("date")
    weights = pd.read_parquet(run_dir / "weights.parquet").set_index("date")
    return summary, daily, weights


@st.cache_data
def load_paper_log():
    if PAPER_LOG.exists():
        return pd.read_parquet(PAPER_LOG)
    return pd.DataFrame()


@st.cache_data
def load_latest_shortlist():
    files = sorted(SHORTLIST_DIR.glob("shortlist_*.parquet"), reverse=True)
    if not files:
        return None
    return pd.read_parquet(files[0])


@st.cache_data
def load_sentiment_daily():
    p = ROOT / "data" / "processed" / "sentiment_daily.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ────────────────────────────────────────────────────────────────────
# Sayfa: ÖZET
# ────────────────────────────────────────────────────────────────────
if page == "🏠 Özet":
    st.title("BIST Sentiment Bot — Sistem Özeti")

    summary, daily, weights = load_latest_backtest()
    shortlist = load_latest_shortlist()
    paper = load_paper_log()

    col1, col2, col3, col4 = st.columns(4)
    if summary:
        pm = summary["portfolio_metrics"]
        bm = summary["benchmark_metrics"]
        sharpe_adj = pm["sharpe"] * 0.85
        cagr_adj = pm["cagr"] * 0.90
        col1.metric("Sharpe (bias-adj)", f"{sharpe_adj:.2f}", f"raw {pm['sharpe']:.2f}")
        col2.metric("CAGR (bias-adj)", f"{cagr_adj*100:.1f}%", f"raw {pm['cagr']*100:.1f}%")
        col3.metric("Max DD", f"{pm['max_dd']*100:.1f}%")
        col4.metric("XU100 BH Sharpe", f"{bm['sharpe']:.2f}", "aynı dönem")
    else:
        st.warning("Henüz backtest yok. Yerelde: `python main.py backtest`")

    st.markdown("---")
    st.subheader("Sistem Durumu")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Veri**")
        n_prices = len(list((ROOT / "data" / "raw" / "prices").glob("*.parquet")))
        n_news = len(list((ROOT / "data" / "raw" / "news").glob("*.parquet")))
        st.write(f"- Fiyat: {n_prices} ticker")
        st.write(f"- Haber: {n_news} günlük snapshot")
        st.write(f"- Paper trade: {len(paper)} kayıt")
        if shortlist is not None:
            st.write(f"- Son shortlist: {len(shortlist)} ticker")
    with c2:
        st.markdown("**Honest Disclosure**")
        st.warning(
            "⚠️ **Survivorship bias**: bugünkü BIST 30 listesi tarihsel olarak "
            "uygulanıyor. Gerçek tarihsel performans **%5-15 daha düşük** olabilir."
        )
        st.info(
            "📐 **Realistic execution**: backtest next-day open simülasyonu "
            "kullanır (close-to-close değil). Overnight gap riski simüle edilir."
        )


# ────────────────────────────────────────────────────────────────────
# Sayfa: BACKTEST
# ────────────────────────────────────────────────────────────────────
elif page == "📈 Backtest":
    st.title("Backtest İncelemesi")
    summary, daily, weights = load_latest_backtest()
    if summary is None:
        st.error("Backtest sonucu yok. `python main.py backtest` çalıştır.")
        st.stop()

    st.caption(f"Run ID: `{summary['run_id']}` · {summary['n_folds']} fold · {summary['n_days']} test günü")

    # Equity curve
    st.subheader("Equity Curve")
    eq_df = daily[["portfolio_equity", "benchmark_equity"]].rename(
        columns={"portfolio_equity": "Portföy", "benchmark_equity": "XU100 BH"}
    )
    st.line_chart(eq_df)

    # Metrics table
    st.subheader("Metrikler")
    pm = summary["portfolio_metrics"]
    bm = summary["benchmark_metrics"]
    metrics_df = pd.DataFrame({
        "Portföy": pm,
        "XU100 BH": bm,
    }).T
    st.dataframe(metrics_df, use_container_width=True)

    # Aylık return
    st.subheader("Aylık Return")
    monthly = daily["portfolio_return"].groupby(pd.Grouper(freq="ME")).apply(
        lambda s: (1 + s).prod() - 1
    )
    monthly_df = pd.DataFrame({"return": monthly})
    monthly_df["renk"] = monthly_df["return"].apply(lambda x: "green" if x > 0 else "red")
    st.bar_chart(monthly_df["return"])

    # En iyi/kötü günler
    st.subheader("Aşırı Günler")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**En iyi 5 gün**")
        top = daily["portfolio_return"].nlargest(5)
        st.dataframe(pd.DataFrame({"return": top}), use_container_width=True)
    with col2:
        st.markdown("**En kötü 5 gün**")
        worst = daily["portfolio_return"].nsmallest(5)
        st.dataframe(pd.DataFrame({"return": worst}), use_container_width=True)

    # Per-ticker katkı
    st.subheader("Pozisyon Sıklığı (top 15)")
    ticker_freq = (weights > 0).sum(axis=0).sort_values(ascending=False).head(15)
    st.bar_chart(ticker_freq)

    # Fold breakdown
    st.subheader("Fold Breakdown")
    fold_df = pd.DataFrame(summary["fold_metrics"])
    st.dataframe(fold_df, use_container_width=True)

    # Bias adjustment slider
    st.subheader("Survivorship Bias Düzeltme")
    bias = st.slider("Bias factor (1.0 = düzeltme yok, 0.85 = standart)", 0.5, 1.0, 0.85, 0.05)
    st.write(f"Sharpe: {pm['sharpe'] * bias:.2f} (raw: {pm['sharpe']:.2f})")
    st.write(f"CAGR: {pm['cagr'] * bias * 100:.1f}% (raw: {pm['cagr']*100:.1f}%)")


# ────────────────────────────────────────────────────────────────────
# Sayfa: SHORTLIST
# ────────────────────────────────────────────────────────────────────
elif page == "🎯 Bugünün Shortlist'i":
    st.title("Günlük Shortlist")
    shortlist = load_latest_shortlist()
    if shortlist is None:
        st.warning("Shortlist yok. `python main.py predict` çalıştır.")
        st.stop()
    st.dataframe(shortlist, use_container_width=True)
    st.caption("Skor: model predict_proba (0-1). Top N en yüksek skor.")
    st.warning("⚠️ Yatırım tavsiyesi değildir. Her kararı kendin doğrula.")


# ────────────────────────────────────────────────────────────────────
# Sayfa: PAPER TRADING
# ────────────────────────────────────────────────────────────────────
elif page == "📝 Paper Trading":
    st.title("Paper Trading — Live ↔ Backtest Cross-check")
    paper = load_paper_log()
    if paper.empty:
        st.warning("Paper log boş. Her `predict` çağrısında otomatik kaydolur.")
        st.stop()

    st.subheader("Özet")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam tahmin", len(paper))
    col2.metric("Unique gün", paper["prediction_date"].nunique())
    col3.metric("5d scored", paper["ret_5d"].notna().sum())

    st.subheader("Tahminler")
    st.dataframe(paper.sort_values("prediction_date", ascending=False), use_container_width=True)

    # Karşılaştırma
    scored_5d = paper.dropna(subset=["ret_5d"])
    if len(scored_5d) > 0:
        st.subheader("5-gün Hold Performansı")
        st.write(f"Ortalama 5g return: {scored_5d['ret_5d'].mean()*100:.2f}%")
        st.write(f"Hit rate: {(scored_5d['ret_5d'] > 0).mean()*100:.1f}%")
        st.write(f"Std: {scored_5d['ret_5d'].std()*100:.2f}%")
        if len(scored_5d) >= 5:
            sharpe_paper = scored_5d['ret_5d'].mean() / scored_5d['ret_5d'].std() * (252/5)**0.5
            st.write(f"Paper Sharpe (annualized): {sharpe_paper:.2f}")


# ────────────────────────────────────────────────────────────────────
# Sayfa: SENTIMENT
# ────────────────────────────────────────────────────────────────────
elif page == "💬 Sentiment":
    st.title("Sentiment Analizi")
    sent = load_sentiment_daily()
    if sent.empty:
        st.warning("Sentiment data yok. Cron `ingest` + `score` + `aggregate` çalıştırınca birikir.")
        st.markdown("""
        **Pipeline:**
        1. `python main.py ingest` — RSS'lerden haber toplar
        2. `python main.py score` — Claude Haiku ile JSON sentiment skoru
        3. `python main.py aggregate` — ticker × tarih ağırlıklı agregasyon
        """)
        st.stop()

    st.subheader("Ticker × Tarih Sentiment")
    pivot = sent.pivot(index="date", columns="ticker", values="sentiment_w")
    st.dataframe(pivot, use_container_width=True)
    st.subheader("Per-ticker özet")
    summary_t = sent.groupby("ticker").agg(
        n_days=("sentiment_w", "count"),
        mean_sentiment=("sentiment_w", "mean"),
        std_sentiment=("sentiment_w", "std"),
        total_news=("news_count", "sum"),
    )
    st.dataframe(summary_t.sort_values("total_news", ascending=False), use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# Sayfa: VERİ KALİTESİ
# ────────────────────────────────────────────────────────────────────
elif page == "🔍 Veri Kalitesi":
    st.title("Yfinance Veri Kalitesi Audit")
    import glob
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "prices" / "*.parquet")))
    if not files:
        st.warning("Henüz fiyat verisi yok. `python main.py ingest`")
        st.stop()

    rows = []
    for f in files:
        df = pd.read_parquet(f)
        name = Path(f).stem
        df["date"] = pd.to_datetime(df["date"])
        rets = df["close"].pct_change().abs()
        rows.append({
            "ticker": name,
            "satir": len(df),
            "NaN_close": int(df["close"].isna().sum()),
            "zero_volume_days": int((df["volume"] == 0).sum()),
            "max_jump_pct": float(rets.max() * 100) if not rets.empty else 0,
            "max_date_gap_days": int(df["date"].diff().dt.days.max() or 0),
            "last_date": df["date"].max().date(),
        })
    qa = pd.DataFrame(rows).sort_values("max_jump_pct", ascending=False)
    st.dataframe(qa, use_container_width=True)
    st.info("`max_jump_pct > 25%` = bedelsiz hisse/split şüphesi. Yahoo adjustment hata yapmış olabilir.")


# ────────────────────────────────────────────────────────────────────
# Sayfa: KONFİG
# ────────────────────────────────────────────────────────────────────
elif page == "⚙️ Konfig & Mimari":
    st.title("Konfigürasyon ve Mimari")

    st.subheader("settings.yaml")
    st.json(SETTINGS)

    st.subheader("Mimari")
    st.markdown("""
```
ingestion/    price_collector, news_collector, data_quality
nlp/          sentiment_claude (Claude API), sentiment_aggregator (6-katmanlı koruma)
features/    loader (parquet→panel), feature_engineering (14 fiyat + 8 sentiment)
models/      triple_barrier (LdP labeling), train (LightGBM)
backtest/    walk_forward (purged+embargoed), engine (next-day open), costs, metrics, survivorship
strategy/    shortlist
reports/    console, paper_trading
dashboard/  app.py (BU SAYFA)
```
""")
    st.subheader("Honest Disclosure Listesi")
    st.warning("""
- **Survivorship bias**: bugünkü BIST 30 listesi tarihsel uygulanıyor
- **Next-day open execution**: gerçekçi, overnight gap riski simüle ediliyor
- **Sentiment NaN**: data yokken 0 değil NaN (LightGBM handles missing)
- **Fold sınırı**: pozisyon ffill fold'lar arası RESETLENİYOR
- **PnL alignment**: weights[t] × rets[t+1] (1-gün shift düzeltildi)
- **Realistik cost**: %0.05 komisyon + %5 BSMV + 15bp spread + 15bp slippage = ~25bp
- **Test coverage**: 33/33 (parity testleri + halisünasyon defansları)
""")
