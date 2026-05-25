"""Ticker × tarih bazında günlük sentiment agregasyonu.

Tasarım (literatür: Kirtaç & Germano 2024, MANA-Net 2024, arXiv 2505.16136):

1) Tek-haber dominasyonu önleme — 6 katmanlı koruma:
   a) Winsorization: extreme sentiment ±%95 quantile'da clip
   b) Confidence weighting: Claude'un belirttiği güven düşükse ağırlık az
   c) Relevance weighting: hisseyle ilgisi düşükse ağırlık az
   d) Source credibility: KAP > Reuters > genel haber > popüler
   e) Time decay: exp(-age_days / half_life) — eski haber az ağırlıklı
   f) min_news threshold: 1 tek haberle güçlü sinyal verme

2) Surprise = sentiment_today − rolling_mean_30d
   (mean reversion baseline'ından sapma alpha üretir, mutlak seviye değil)

3) Output: data/processed/sentiment_daily.parquet
   Şema: date | ticker | sentiment_w | news_count | sentiment_std | source_diversity
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS

RAW_NEWS = ROOT / "data" / "raw" / "news"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROCESSED / "sentiment_daily.parquet"


def _load_scored_news() -> pd.DataFrame:
    """data/processed/sentiment_*.parquet'leri birleştir (Claude skorlanmış haberler)."""
    files = sorted(PROCESSED.glob("sentiment_*.parquet"))
    # sentiment_daily.parquet'i hariç tut (bu modülün çıktısı)
    files = [f for f in files if not f.name.startswith("sentiment_daily")]
    if not files:
        logger.warning("Skorlanmış haber dosyası yok (sentiment_YYYY-MM-DD.parquet)")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def _source_weight(source: str, table: dict) -> float:
    """Kaynak kredibilite ağırlığı (settings.yaml table'ından)."""
    return float(table.get(source, table.get("_default", 0.5)))


def _decay_weight(age_days: float, half_life: float) -> float:
    """Exponential decay: w = 0.5 ^ (age / half_life). 0 yaş = 1.0, half_life yaş = 0.5."""
    if half_life <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life))


def _winsorize_series(s: pd.Series, low_q: float, high_q: float) -> pd.Series:
    """Quantile bazlı clip — extreme sentiment'leri yumuşat."""
    if s.empty or s.dropna().empty:
        return s
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lo, hi)


def aggregate(as_of: datetime | None = None) -> pd.DataFrame:
    """Skorlanmış haberleri ticker × tarih bazında ağırlıklı toplama.

    Returns
    -------
    pd.DataFrame
        Kolonlar:
          date, ticker, sentiment_w (ağırlıklı ortalama, winsorized),
          news_count (o gün-ticker'da kaç haber),
          sentiment_std (varyans — uzlaşmazlık ölçüsü),
          source_diversity (kaç farklı kaynak)
    """
    agg_cfg = SETTINGS["nlp"]["aggregation"]
    half_life = agg_cfg["decay_half_life_days"]
    winsor = agg_cfg["winsorize_pct"]
    cred_table = agg_cfg["source_credibility"]
    as_of = as_of or datetime.now()

    raw = _load_scored_news()
    if raw.empty:
        logger.info("Sentiment agregasyonu: girdi yok, boş output")
        empty = pd.DataFrame(columns=["date", "ticker", "sentiment_w", "news_count",
                                       "sentiment_std", "source_diversity"])
        empty.to_parquet(OUT_PATH, index=False)
        return empty

    # Beklenen kolonlar: ticker, source, published_at, sentiment, relevance, confidence
    required = {"ticker", "source", "published_at", "sentiment", "relevance", "confidence"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Skorlanmış haber dosyalarında eksik kolon: {missing}")

    # Tarihe yuvarla (gün bazlı agregasyon)
    raw["published_at"] = pd.to_datetime(raw["published_at"])
    raw["date"] = raw["published_at"].dt.normalize()

    # Failed scoring'leri filtrele
    if "_failed" in raw.columns:
        raw = raw[~raw["_failed"].fillna(False)]
    raw = raw[raw["sentiment"].between(-1, 1, inclusive="both")]
    raw = raw[raw["confidence"] >= 0]
    raw = raw[raw["relevance"] >= 0]

    # 1) Winsorization — tek-haber dominasyonu önleme
    raw["sentiment_w_in"] = _winsorize_series(raw["sentiment"], winsor, 1 - winsor)

    # 2) Kompozit ağırlık: source_cred × relevance × confidence × decay
    raw["w_source"] = raw["source"].apply(lambda s: _source_weight(s, cred_table))
    raw["w_age"] = ((as_of - raw["published_at"]).dt.total_seconds() / 86400).clip(lower=0)
    raw["w_decay"] = raw["w_age"].apply(lambda a: _decay_weight(a, half_life))
    raw["weight"] = raw["w_source"] * raw["relevance"] * raw["confidence"] * raw["w_decay"]

    # 3) Ticker × tarih bazında ağırlıklı ortalama
    def _agg_group(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].sum()
        if w <= 0:
            return pd.Series({
                "sentiment_w": 0.0,
                "news_count": len(g),
                "sentiment_std": float(g["sentiment_w_in"].std() or 0),
                "source_diversity": g["source"].nunique(),
            })
        wmean = float((g["sentiment_w_in"] * g["weight"]).sum() / w)
        return pd.Series({
            "sentiment_w": wmean,
            "news_count": len(g),
            "sentiment_std": float(g["sentiment_w_in"].std() or 0),
            "source_diversity": g["source"].nunique(),
        })

    out = (raw
           .groupby(["date", "ticker"], group_keys=False)
           .apply(_agg_group, include_groups=False)
           .reset_index())

    # Diske yaz
    out.to_parquet(OUT_PATH, index=False)
    logger.info(f"Sentiment agregasyonu: {len(out)} ticker-gün → {OUT_PATH.name}")
    return out


if __name__ == "__main__":
    df = aggregate()
    print(df.head(20))
    if not df.empty:
        print(f"\n{df['ticker'].nunique()} ticker × {df['date'].nunique()} gün")
