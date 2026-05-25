"""Claude Haiku ile per-headline sentiment skorlama.

Her haber için JSON döner:
  {
    "sentiment": -1.0 .. +1.0,
    "relevance": 0.0 .. 1.0,
    "confidence": 0.0 .. 1.0,
    "category": "earnings" | "macro" | "rumor" | "guidance" | "M&A" | "other",
    "rationale": "kısa açıklama"
  }

Tüm yanıtlar diskcache (content hash) ile cache'lenir — aynı metin
tekrar skorlanmaz. Bu Claude API faturasını çok düşürür.

Kullanım
--------
    from nlp.sentiment_claude import score_news
    df = pd.read_parquet("data/raw/news/news_2026-05-22.parquet")
    scored = score_news(df)   # df'e sentiment kolonları ekler
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pandas as pd
from anthropic import Anthropic
from diskcache import Cache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import ROOT, SETTINGS, env

_cache = Cache(str(ROOT / SETTINGS["nlp"]["cache_dir"]))

# Sürekli aynı clienti reuse et (lazy init)
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = env("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment var ayarlı değil")
        _client = Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = """Sen finansal sentiment analizi yapan bir asistansın.
Sana bir Türkçe finans haberi başlığı + özeti verilecek. Bu haberin
ilgili BIST hissesi için kısa vadeli (1-5 işlem günü) fiyat etkisine dair
sentiment skorlaması yapacaksın.

ÇIKTI: Sadece geçerli JSON, başka metin YOK. Şu alanlarla:
- sentiment: -1.0 (çok olumsuz) .. +1.0 (çok olumlu) float
- relevance: 0.0 (alakasız) .. 1.0 (doğrudan hisseyi etkiler) float
- confidence: 0.0 (belirsiz) .. 1.0 (kesin) float
- category: ["earnings", "guidance", "macro", "M&A", "rumor", "regulation", "operational", "other"] arasından bir
- rationale: 1 cümle gerekçe (Türkçe)

KURALLAR:
- "Şirket bedelsiz hisse dağıtacak" → genelde +0.3 (dilution riski ama sinyal)
- "Karlılık zayıfladı / beklenti altı" → -0.5 ile -0.9 arası
- "Stratejik anlaşma / büyük ihale aldı" → +0.5 ile +0.9 arası
- "Genel piyasa yorumu, hisse özelinde değil" → relevance < 0.3
- "Söylenti / dedikodu" → confidence < 0.4
- Tarihsel/eski olay → relevance < 0.5
- Bir şüphen varsa MUHAFAZAKAR davran (düşük confidence)."""


def _content_hash(ticker: str, text: str) -> str:
    """Stable cache key: ticker + text birlikte hash."""
    payload = (ticker + "||" + text.strip()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def _ask_claude(ticker: str, title: str, summary: str) -> dict[str, Any]:
    """Claude'a tek bir haber gönderir, JSON parse eder."""
    user = f"""TICKER: {ticker}
BAŞLIK: {title}
ÖZET: {summary}

Bu haberin {ticker} hissesi için kısa vadeli sentiment'ini JSON olarak değerlendir."""

    client = _get_client()
    resp = client.messages.create(
        model=SETTINGS["nlp"]["claude_model"],
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text.strip()
    # Claude bazen "```json ... ```" wrapper koyar — temizle
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse fail: {text[:200]}")
        raise
    # Schema sanity check
    for k in ("sentiment", "relevance", "confidence", "category"):
        if k not in data:
            raise ValueError(f"Eksik alan {k}: {data}")
    return data


def score_one(ticker: str, title: str, summary: str = "") -> dict[str, Any]:
    """Cache'li tek haber skorlaması."""
    text = (title + " " + summary).strip()
    key = _content_hash(ticker, text)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        result = _ask_claude(ticker, title, summary)
    except Exception as e:
        logger.error(f"[{ticker}] Claude fail → {e}")
        result = {
            "sentiment": 0.0, "relevance": 0.0, "confidence": 0.0,
            "category": "other", "rationale": f"FAILED: {e}",
            "_failed": True,
        }
    _cache.set(key, result)
    return result


def score_news(df: pd.DataFrame, throttle_ms: int = 100) -> pd.DataFrame:
    """Bir news DataFrame için tüm satırları skorlar.

    df: id, ticker, title, summary kolonlarına sahip olmalı.
    Çıktı: df + ['sentiment','relevance','confidence','category','rationale']
    """
    if df.empty:
        return df

    cols = ("sentiment", "relevance", "confidence", "category", "rationale")
    results: list[dict] = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        ticker = getattr(row, "ticker", "")
        title = getattr(row, "title", "")
        summary = getattr(row, "summary", "")
        res = score_one(ticker, title, summary)
        results.append(res)
        if (i + 1) % 20 == 0:
            logger.info(f"Sentiment scoring: {i+1}/{n}")
        if throttle_ms > 0:
            time.sleep(throttle_ms / 1000)

    add = pd.DataFrame(results)[list(cols)]
    out = pd.concat([df.reset_index(drop=True), add], axis=1)
    logger.info(f"Sentiment scoring bitti — {n} kayıt")
    return out


if __name__ == "__main__":
    # Quick demo: bir kayıtla test et
    demo = score_one(
        "ASELS.IS",
        "ASELSAN savunma bakanlığından 500 milyon dolarlık ihale aldı",
        "ASELSAN bugün KAP'a yaptığı açıklamada Türk Silahlı Kuvvetleri için "
        "500 milyon dolar değerinde radar sistemi sözleşmesi imzaladığını duyurdu.",
    )
    print(json.dumps(demo, ensure_ascii=False, indent=2))
