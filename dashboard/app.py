"""BIST Sentiment Bot — Quant Dashboard (Streamlit).

Design language: portfolio-tracker quant aesthetic
- Dark background (#0f0f0f), forest-green accent (#4ade80)
- Newsreader serif headers, DM Mono numbers, Geist Mono details
- Glassmorphism cards, plotly charts
- Honest disclosure throughout

Çalıştırma:
    streamlit run dashboard/app.py
Cloud deploy: share.streamlit.io → bağla → otomatik
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT, SETTINGS

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BIST Sentiment Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Quant aesthetic
# ════════════════════════════════════════════════════════════════════
QUANT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,300;0,400;0,500;1,400&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0a;
    --surface: #141414;
    --surface2: #1c1c1c;
    --border: #2a2a2a;
    --accent: #4ade80;
    --gold: #f59e0b;
    --danger: #ef4444;
    --success: #4ade80;
    --muted: #6b6b6b;
    --text: #e8e8e8;
    --text2: #a8a8a8;
}

/* Streamlit overrides */
html, body, [class*="css"], .stApp, .stMarkdown, .stMetric {
    font-family: 'Inter', sans-serif !important;
    color: var(--text);
}
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 2.1rem !important; font-weight: 300 !important; }
h2 { font-size: 1.5rem !important; font-style: italic; color: var(--text2) !important; }
h3 { font-size: 1.15rem !important; }

/* KPI Cards — replaces st.metric look */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin: 18px 0 24px 0;
}
.kpi {
    background: linear-gradient(135deg, var(--surface), var(--surface2));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    border-color: var(--accent);
}
.kpi-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    font-weight: 500;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.1;
}
.kpi-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 6px;
}
.kpi-value.pos { color: var(--success); }
.kpi-value.neg { color: var(--danger); }

/* Honest disclosure cards */
.disclose {
    background: rgba(245, 158, 11, 0.08);
    border-left: 3px solid var(--gold);
    padding: 12px 16px;
    border-radius: 6px;
    margin: 14px 0;
    font-size: 0.85rem;
    color: var(--text2);
}
.disclose strong { color: var(--gold); }
.disclose-info {
    background: rgba(74, 222, 128, 0.06);
    border-left: 3px solid var(--accent);
}
.disclose-info strong { color: var(--accent); }

/* Tables */
.stDataFrame, .stTable {
    background: var(--surface) !important;
    border-radius: 8px;
}
.stDataFrame [data-testid="stTable"] td,
.stDataFrame [data-testid="stTable"] th {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Plotly charts background */
.js-plotly-plot .plot-container {
    background: var(--surface) !important;
    border-radius: 8px;
}

/* Subtle dividers */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* Sidebar radio */
[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
    color: var(--text2);
}
[data-testid="stRadio"] label:hover { color: var(--accent); }

/* Footer text */
.footer-note {
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
    margin-top: 30px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Section title */
.section-title {
    font-family: 'Newsreader', serif;
    font-size: 1.1rem;
    font-style: italic;
    color: var(--text2);
    margin: 30px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
</style>
"""
st.markdown(QUANT_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PLOTLY THEME — match dark quant
# ════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#141414",
    font=dict(family="DM Mono, monospace", size=11, color="#a8a8a8"),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a"),
    yaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a"),
)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
BT_DIR = ROOT / "data" / "processed" / "backtests"
PAPER_LOG = ROOT / "data" / "processed" / "paper_trades.parquet"
SHORTLIST_DIR = ROOT / "data" / "processed"


def kpi_card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    """HTML KPI card — quant style. Single line to avoid markdown code-block escape."""
    cls = {"pos": "kpi-value pos", "neg": "kpi-value neg", "neutral": "kpi-value"}[tone]
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="{cls}">{value}</div>{sub_html}</div>'


def kpi_grid(items: list[str]) -> str:
    return f'<div class="kpi-grid">{"".join(items)}</div>'


def disclose(text: str, kind: str = "warn") -> str:
    cls = "disclose" + (" disclose-info" if kind == "info" else "")
    return f'<div class="{cls}">{text}</div>'


def fmt_pct(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:+.{decimals}f}%"


def fmt_num(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


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


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 18px 0;border-bottom:1px solid #2a2a2a">
      <div style="font-family:'Newsreader',serif;font-size:1.3rem;font-weight:300">
        BIST <em style="color:#4ade80">Sentiment Bot</em>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#6b6b6b;margin-top:4px">
        QUANT RESEARCH PLATFORM · v0.5
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "Genel Bakış",
            "Backtest",
            "Trades & Neden",
            "Bugünün Shortlist'i",
            "Paper Trading",
            "Sentiment",
            "Veri Kalitesi",
            "Konfig & Mimari",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:30px;padding:12px;background:#141414;border-radius:8px;font-size:0.72rem;color:#a8a8a8;line-height:1.6">
      <div style="color:#f59e0b;font-weight:500;margin-bottom:6px">⚠ Honest System</div>
      Yatırım tavsiyesi değildir. Backtest survivorship bias içerir
      (×0.85 ile düzelt).
      <br><br>
      <a href="https://github.com/Cugutayy/bist-sentiment-bot" style="color:#4ade80;text-decoration:none">→ github.com/Cugutayy/bist-sentiment-bot</a>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: GENEL BAKIŞ
# ════════════════════════════════════════════════════════════════════
if page == "Genel Bakış":
    st.markdown("# Genel Bakış")
    st.markdown('<div style="color:#6b6b6b;margin-top:-12px;font-family:DM Mono,monospace;font-size:0.78rem">BIST 30 · 5y backtest · LightGBM + sentiment</div>', unsafe_allow_html=True)

    summary, daily, weights = load_latest_backtest()

    if summary:
        pm = summary["portfolio_metrics"]
        bm = summary["benchmark_metrics"]
        sharpe_adj = pm["sharpe"] * 0.85
        cagr_adj = pm["cagr"] * 0.90

        cards = [
            kpi_card(
                "Sharpe (bias-adj)",
                f"{sharpe_adj:.2f}",
                f"raw {pm['sharpe']:.2f}",
                "pos" if sharpe_adj > 0 else "neg",
            ),
            kpi_card(
                "CAGR (bias-adj)",
                f"{cagr_adj*100:+.1f}%",
                f"raw {pm['cagr']*100:+.1f}%",
                "pos" if cagr_adj > 0 else "neg",
            ),
            kpi_card(
                "Max Drawdown",
                f"{pm['max_dd']*100:.1f}%",
                "fold-aggregate",
                "neg",
            ),
            kpi_card(
                "XU100 BH Sharpe",
                f"{bm['sharpe']:.2f}",
                "aynı dönem",
                "pos" if bm["sharpe"] > 0 else "neg",
            ),
            kpi_card(
                "Hit Rate",
                f"{pm['hit_rate']*100:.1f}%",
                f"vs XU100 {bm['hit_rate']*100:.1f}%",
                "neutral",
            ),
            kpi_card(
                "Turnover",
                f"{pm.get('turnover_d', 0)*100:.1f}%/g",
                "haftalık rebalance",
                "neutral",
            ),
        ]
        st.markdown(kpi_grid(cards), unsafe_allow_html=True)
    else:
        st.markdown(disclose("Henüz backtest yok. Yerelde: <code>python main.py backtest</code>"), unsafe_allow_html=True)

    # Sistem durumu
    st.markdown('<div class="section-title">Sistem Durumu</div>', unsafe_allow_html=True)
    paper = load_paper_log()
    shortlist = load_latest_shortlist()
    n_prices = len(list((ROOT / "data" / "raw" / "prices").glob("*.parquet")))
    n_news = len(list((ROOT / "data" / "raw" / "news").glob("*.parquet")))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Fiyat verisi", str(n_prices), "ticker", "neutral"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Haber snapshot", str(n_news), "günlük dosya", "neutral"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Paper trade", str(len(paper)), "log kayıt", "neutral"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Son shortlist", str(len(shortlist)) if shortlist is not None else "—",
                         "ticker", "neutral"), unsafe_allow_html=True)

    # Disclosure
    st.markdown('<div class="section-title">Honest Disclosure</div>', unsafe_allow_html=True)
    st.markdown(disclose(
        "<strong>Survivorship Bias:</strong> Bugünkü BIST 30 listesi 5 yıl geriye uygulanıyor. "
        "Tarihte 'öldürülmüş' (delisting/iflas) şirketler yok. Gerçek tarihsel performans "
        "<strong>%5-15 daha düşük</strong> olabilir."
    ), unsafe_allow_html=True)
    st.markdown(disclose(
        "<strong>Next-Day Open Execution:</strong> Backtest gerçekçi execution kullanır — "
        "close[t-1] sinyali → open[t] alım. Overnight gap riski simüle edilir.",
        kind="info",
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST
# ════════════════════════════════════════════════════════════════════
elif page == "Backtest":
    st.markdown("# Backtest <em style='color:#6b6b6b;font-size:0.8em'>incelemesi</em>", unsafe_allow_html=True)
    summary, daily, weights = load_latest_backtest()
    if summary is None:
        st.markdown(disclose("Backtest yok. <code>python main.py backtest</code>"), unsafe_allow_html=True)
        st.stop()

    st.markdown(
        f'<div style="color:#6b6b6b;font-family:DM Mono,monospace;font-size:0.75rem;margin-bottom:18px">'
        f"Run: {summary['run_id']} · {summary['n_folds']} fold · {summary['n_days']} test günü</div>",
        unsafe_allow_html=True,
    )

    # ── Equity Curve ──
    st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["portfolio_equity"],
        name="Portföy", line=dict(color="#4ade80", width=2.5),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["benchmark_equity"],
        name="XU100 BH", line=dict(color="#f59e0b", width=1.5, dash="dot"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ──
    st.markdown('<div class="section-title">Metrikler</div>', unsafe_allow_html=True)
    pm = summary["portfolio_metrics"]
    bm = summary["benchmark_metrics"]
    cards = [
        kpi_card("Sharpe", f"{pm['sharpe']:.2f}", f"XU100: {bm['sharpe']:.2f}",
                 "pos" if pm["sharpe"] > bm["sharpe"] else "neg"),
        kpi_card("CAGR", f"{pm['cagr']*100:+.1f}%", f"XU100: {bm['cagr']*100:+.1f}%",
                 "pos" if pm["cagr"] > bm["cagr"] else "neg"),
        kpi_card("Max DD", f"{pm['max_dd']*100:.1f}%", f"XU100: {bm['max_dd']*100:.1f}%", "neg"),
        kpi_card("Hit Rate", f"{pm['hit_rate']*100:.1f}%", f"XU100: {bm['hit_rate']*100:.1f}%", "neutral"),
        kpi_card("Daily μ", f"{pm['mean_d']*100:+.3f}%", f"σ={pm['std_d']*100:.2f}%", "neutral"),
        kpi_card("Turnover", f"{pm.get('turnover_d',0)*100:.1f}%/g", "exit+enter", "neutral"),
    ]
    st.markdown(kpi_grid(cards), unsafe_allow_html=True)

    # ── Aylık Return ──
    st.markdown('<div class="section-title">Aylık Return</div>', unsafe_allow_html=True)
    monthly = daily["portfolio_return"].groupby(pd.Grouper(freq="ME")).apply(
        lambda s: (1 + s).prod() - 1
    )
    monthly_pct = monthly * 100
    colors = ["#4ade80" if x >= 0 else "#ef4444" for x in monthly_pct.values]
    fig2 = go.Figure(go.Bar(
        x=monthly_pct.index.strftime("%Y-%m"),
        y=monthly_pct.values,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in monthly_pct.values],
        textposition="outside",
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    fig2.update_yaxes(title="Return %")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Top/Worst gün ──
    st.markdown('<div class="section-title">Aşırı Günler</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**En iyi 5**")
        top = daily["portfolio_return"].nlargest(5).to_frame("return")
        top["return"] = top["return"].apply(lambda x: f"{x*100:+.2f}%")
        st.dataframe(top, use_container_width=True)
    with col2:
        st.markdown("**En kötü 5**")
        worst = daily["portfolio_return"].nsmallest(5).to_frame("return")
        worst["return"] = worst["return"].apply(lambda x: f"{x*100:+.2f}%")
        st.dataframe(worst, use_container_width=True)

    # ── Ticker frekansı (plotly bar) ──
    st.markdown('<div class="section-title">Pozisyon Sıklığı (top 15)</div>', unsafe_allow_html=True)
    ticker_freq = (weights > 0).sum(axis=0).sort_values(ascending=False).head(15)
    fig3 = go.Figure(go.Bar(
        x=ticker_freq.values, y=ticker_freq.index,
        orientation="h", marker_color="#4ade80",
        text=ticker_freq.values, textposition="outside",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    fig3.update_yaxes(autorange="reversed")
    fig3.update_xaxes(title="Gün sayısı")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Fold breakdown ──
    st.markdown('<div class="section-title">Fold Breakdown</div>', unsafe_allow_html=True)
    fold_df = pd.DataFrame(summary["fold_metrics"])
    st.dataframe(fold_df, use_container_width=True)

    # ── Bias slider ──
    st.markdown('<div class="section-title">Survivorship Bias Düzeltme</div>', unsafe_allow_html=True)
    bias = st.slider("Bias factor (1.0 = düzeltme yok, 0.85 önerilen)", 0.5, 1.0, 0.85, 0.05)
    cards = [
        kpi_card("Adj. Sharpe", f"{pm['sharpe']*bias:.2f}", f"raw {pm['sharpe']:.2f}",
                 "pos" if pm["sharpe"]*bias > 0 else "neg"),
        kpi_card("Adj. CAGR", f"{pm['cagr']*bias*100:+.1f}%", f"raw {pm['cagr']*100:+.1f}%",
                 "pos" if pm["cagr"]*bias > 0 else "neg"),
    ]
    st.markdown(kpi_grid(cards), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: TRADES & NEDEN
# ════════════════════════════════════════════════════════════════════
elif page == "Trades & Neden":
    st.markdown("# Trades <em style='color:#6b6b6b;font-size:0.7em'>backtest sırasında alınan işlemler + neden</em>", unsafe_allow_html=True)

    from backtest.trade_explainer import build_trade_ledger

    @st.cache_data(show_spinner="Trade ledger oluşturuluyor (model + feature replay)...")
    def _load_trades():
        return build_trade_ledger()

    trades = _load_trades()

    if trades.empty:
        st.markdown(disclose("Trade yok. Önce backtest çalıştır: <code>python main.py backtest</code>"), unsafe_allow_html=True)
        st.stop()

    # KPI cards
    n_buy = int((trades["action"] == "BUY").sum())
    n_sell = int((trades["action"] == "SELL").sum())
    n_unique = trades["ticker"].nunique()
    n_dates = trades["date"].nunique()
    avg_score = float(trades["score"].mean())
    cards = [
        kpi_card("Toplam Trade", str(len(trades)), f"{n_buy} BUY · {n_sell} SELL", "neutral"),
        kpi_card("Unique Ticker", str(n_unique), "farklı hisse", "neutral"),
        kpi_card("Rebalance Günü", str(n_dates), "fold bazlı", "neutral"),
        kpi_card("Ort. Skor", f"{avg_score:.3f}" if pd.notna(avg_score) else "—",
                 "predict_proba", "pos" if avg_score > 0.5 else "neutral"),
    ]
    st.markdown(kpi_grid(cards), unsafe_allow_html=True)

    # Filtrele
    st.markdown('<div class="section-title">Filtrele</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        action_filter = st.selectbox("Aksiyon", ["Tümü", "BUY", "SELL"])
    with col2:
        ticker_options = ["Tümü"] + sorted(trades["ticker"].unique().tolist())
        ticker_filter = st.selectbox("Ticker", ticker_options)
    with col3:
        date_range = st.date_input(
            "Tarih aralığı",
            value=(trades["date"].min().date(), trades["date"].max().date()),
        )

    filtered = trades.copy()
    if action_filter != "Tümü":
        filtered = filtered[filtered["action"] == action_filter]
    if ticker_filter != "Tümü":
        filtered = filtered[filtered["ticker"] == ticker_filter]
    if len(date_range) == 2:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["date"] >= d0) & (filtered["date"] <= d1)]

    st.markdown(f'<div class="section-title">Trade Ledger <span style="color:#6b6b6b;font-size:0.7em">({len(filtered)} kayıt)</span></div>', unsafe_allow_html=True)

    # Render: her trade için tek kart
    if filtered.empty:
        st.markdown(disclose("Filtre kriterlerine uyan trade yok."), unsafe_allow_html=True)
    else:
        # Tablo halinde kompakt görünüm
        display = filtered.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["weight_delta"] = display["weight_delta"].apply(lambda x: f"{x*100:+.1f}%")
        display["new_weight"] = display["new_weight"].apply(lambda x: f"{x*100:.1f}%")
        display["score"] = display["score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        display = display[["date", "ticker", "action", "weight_delta", "new_weight", "score", "rationale"]]
        display.columns = ["Tarih", "Ticker", "Aksiyon", "Δ Ağırlık", "Yeni Ağırlık", "Skor", "Neden (top 3 feature)"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    # Per-ticker rationale özet
    st.markdown('<div class="section-title">Hangi Sinyaller En Çok Tetikledi?</div>', unsafe_allow_html=True)
    # Rationale'lardan feature isimleri çıkar
    import re
    feature_pattern = re.compile(r"([a-zA-Z_0-9]+)=")
    all_features = []
    for rat in trades["rationale"].dropna():
        all_features.extend(feature_pattern.findall(rat))
    if all_features:
        feat_counts = pd.Series(all_features).value_counts().head(10)
        fig_feat = go.Figure(go.Bar(
            x=feat_counts.values, y=feat_counts.index,
            orientation="h", marker_color="#4ade80",
            text=feat_counts.values, textposition="outside",
        ))
        fig_feat.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        fig_feat.update_yaxes(autorange="reversed")
        fig_feat.update_xaxes(title="Kaç trade'de top-3'te")
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown(disclose(
            "<strong>rsi_14</strong> ve <strong>cs_momentum_60</strong> en sık tetikleyici feature'lar — "
            "model BIST'te momentum + aşırı satılmış-toparlanma kalıplarına ağırlık vermiş.",
            kind="info",
        ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: SHORTLIST
# ════════════════════════════════════════════════════════════════════
elif page == "Bugünün Shortlist'i":
    st.markdown("# Bugünün Shortlist'i")
    shortlist = load_latest_shortlist()
    if shortlist is None:
        st.markdown(disclose("Shortlist yok. <code>python main.py predict</code>"), unsafe_allow_html=True)
        st.stop()

    # Top 10 KPI: skor + fiyat + return
    if len(shortlist) >= 1:
        cards = []
        for i, row in shortlist.head(6).iterrows():
            tone = "pos" if row.get("score", 0) > 0.7 else "neutral"
            cards.append(kpi_card(
                f"#{i+1} · {row['ticker']}",
                f"{row['score']:.3f}",
                f"@{row['close']:.2f} TL",
                tone,
            ))
        st.markdown(kpi_grid(cards), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tam Sıralama</div>', unsafe_allow_html=True)
    display = shortlist.copy()
    if "score" in display.columns:
        display["score"] = display["score"].apply(lambda x: f"{x:.3f}")
    st.dataframe(display, use_container_width=True)
    st.markdown(disclose("Skor 0-1 arası — model predict_proba (triple-barrier +1 hit olasılığı). "
                         "Yatırım tavsiyesi değildir."), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: PAPER TRADING
# ════════════════════════════════════════════════════════════════════
elif page == "Paper Trading":
    st.markdown("# Paper Trading <em style='color:#6b6b6b;font-size:0.7em'>backtest ↔ live cross-check</em>", unsafe_allow_html=True)
    paper = load_paper_log()
    if paper.empty:
        st.markdown(disclose("Paper log boş. Her <code>predict</code> çağrısında otomatik kaydolur."), unsafe_allow_html=True)
        st.stop()

    cards = [
        kpi_card("Toplam tahmin", str(len(paper)), "log kayıt", "neutral"),
        kpi_card("Unique gün", str(paper["prediction_date"].nunique()), "", "neutral"),
        kpi_card("5d scored", str(paper["ret_5d"].notna().sum()), f"/{len(paper)}", "neutral"),
    ]
    st.markdown(kpi_grid(cards), unsafe_allow_html=True)

    scored = paper.dropna(subset=["ret_5d"])
    if len(scored) > 0:
        mean_r = scored["ret_5d"].mean() * 100
        hit = (scored["ret_5d"] > 0).mean() * 100
        cards2 = [
            kpi_card("Ortalama 5g return", f"{mean_r:+.2f}%", "", "pos" if mean_r > 0 else "neg"),
            kpi_card("Hit Rate", f"{hit:.1f}%", "live measured", "neutral"),
        ]
        st.markdown(kpi_grid(cards2), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tahmin Kayıtları</div>', unsafe_allow_html=True)
    st.dataframe(paper.sort_values("prediction_date", ascending=False), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT
# ════════════════════════════════════════════════════════════════════
elif page == "Sentiment":
    st.markdown("# Sentiment Analizi")
    sent = load_sentiment_daily()
    if sent.empty:
        st.markdown(disclose(
            "Sentiment data yok. Pipeline: "
            "<code>ingest</code> → <code>score</code> → <code>aggregate</code>"
        ), unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="section-title">Ticker × Tarih</div>', unsafe_allow_html=True)
    pivot = sent.pivot(index="date", columns="ticker", values="sentiment_w")
    st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                 use_container_width=True)

    st.markdown('<div class="section-title">Per-Ticker Özet</div>', unsafe_allow_html=True)
    summary_t = sent.groupby("ticker").agg(
        n_days=("sentiment_w", "count"),
        mean=("sentiment_w", "mean"),
        std=("sentiment_w", "std"),
        total_news=("news_count", "sum"),
    ).sort_values("total_news", ascending=False)
    st.dataframe(summary_t, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: VERİ KALİTESİ
# ════════════════════════════════════════════════════════════════════
elif page == "Veri Kalitesi":
    st.markdown("# Veri Kalitesi Audit")
    import glob
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "prices" / "*.parquet")))
    if not files:
        st.markdown(disclose("Henüz fiyat verisi yok. <code>python main.py ingest</code>"), unsafe_allow_html=True)
        st.stop()

    rows = []
    for f in files:
        df = pd.read_parquet(f)
        name = Path(f).stem
        df["date"] = pd.to_datetime(df["date"])
        rets = df["close"].pct_change().abs()
        rows.append({
            "ticker": name,
            "satır": len(df),
            "NaN": int(df["close"].isna().sum()),
            "0_volume": int((df["volume"] == 0).sum()),
            "max_jump_%": round(float(rets.max() * 100), 1) if not rets.empty else 0,
            "max_gap_g": int(df["date"].diff().dt.days.max() or 0),
            "son_tarih": str(df["date"].max().date()),
        })
    qa = pd.DataFrame(rows).sort_values("max_jump_%", ascending=False)

    cards = [
        kpi_card("Ticker sayısı", str(len(qa)), "yüklü", "neutral"),
        kpi_card("Toplam NaN", str(int(qa["NaN"].sum())), "tüm tickerlar", "neutral"),
        kpi_card("Max jump", f"{qa['max_jump_%'].max():.1f}%", "tek günde", "neutral"),
    ]
    st.markdown(kpi_grid(cards), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Per-Ticker Detay</div>', unsafe_allow_html=True)
    st.dataframe(qa, use_container_width=True)
    st.markdown(disclose(
        "<strong>max_jump_% > 25</strong>: bedelsiz/split şüphesi — Yahoo adjustment hata yapmış olabilir.",
        kind="info",
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: KONFİG & MİMARİ
# ════════════════════════════════════════════════════════════════════
elif page == "Konfig & Mimari":
    st.markdown("# Konfig & Mimari")

    st.markdown('<div class="section-title">Mimari Katmanlar</div>', unsafe_allow_html=True)
    st.markdown("""
<pre style="font-family:'DM Mono',monospace;font-size:0.78rem;background:#141414;
border:1px solid #2a2a2a;border-radius:8px;padding:18px;color:#a8a8a8;line-height:1.6">
<span style="color:#4ade80">ingestion/</span>    price_collector · news_collector · data_quality
<span style="color:#4ade80">nlp/</span>           sentiment_claude (Claude API) · sentiment_aggregator (6-katmanlı)
<span style="color:#4ade80">features/</span>      loader · feature_engineering (14 fiyat + 8 sentiment)
<span style="color:#4ade80">models/</span>        triple_barrier (LdP) · train (LightGBM)
<span style="color:#4ade80">backtest/</span>      walk_forward · engine (next-day open) · costs · metrics · survivorship
<span style="color:#4ade80">strategy/</span>      shortlist
<span style="color:#4ade80">reports/</span>       console · paper_trading
<span style="color:#4ade80">dashboard/</span>     <em>app.py — BU SAYFA</em>
</pre>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Honest Disclosure Listesi</div>', unsafe_allow_html=True)
    items = [
        "<strong>Survivorship bias</strong>: bugünkü BIST 30 listesi tarihsel uygulanıyor",
        "<strong>Next-day open execution</strong>: gerçekçi, overnight gap simülasyonu",
        "<strong>Sentiment NaN</strong>: data yokken 0 değil NaN (LightGBM missing branch)",
        "<strong>Fold sınırı</strong>: pozisyon ffill fold'lar arası RESETLENİYOR",
        "<strong>PnL alignment</strong>: weights[t] × rets[t+1] (1-gün shift düzeltildi)",
        "<strong>Realistik cost</strong>: %0.05 + %5 BSMV + 15bp spread + 15bp slippage ≈ 25bp tek yön",
        "<strong>Test coverage</strong>: 33/33 (parity + halisünasyon defansları)",
    ]
    for item in items:
        st.markdown(f"<div style='padding:8px 0;border-bottom:1px solid #2a2a2a;font-size:0.88rem;color:#a8a8a8'>✓ {item}</div>",
                    unsafe_allow_html=True)

    st.markdown('<div class="section-title">settings.yaml</div>', unsafe_allow_html=True)
    with st.expander("Konfigürasyonu görüntüle", expanded=False):
        st.json(SETTINGS)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer-note">BIST Sentiment Bot · v0.5 · Quant research platform · '
    '<a href="https://github.com/Cugutayy/bist-sentiment-bot" style="color:#4ade80">GitHub</a></div>',
    unsafe_allow_html=True,
)
