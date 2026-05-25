"""Türkçe finans haberlerini RSS feed'lerden topla, BIST ticker'a eşle.

Her run:
1) Her feed'i fetch et (try/except + log)
2) Her makale için: hangi BIST ticker'ı geçiyor? (basit isim eşleştirme)
3) data/raw/news/news_YYYY-MM-DD.parquet'a append (dedup by URL)

Notlar
-----
- snscrape öldü, X API ücretli → sosyal medya YOK.
- Bu basit RSS yaklaşımı sentiment için yeterli MVP.
- Faz 1B'de Playwright tabanlı KAP scraper eklenecek (resmi açıklamalar).
"""
from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path


def _utcnow() -> datetime:
    """Naive UTC datetime — pandas/parquet için tz-aware sorun yaratır."""
    return datetime.now(UTC).replace(tzinfo=None)

import feedparser
import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS

RAW_NEWS_DIR = ROOT / "data" / "raw" / "news"
RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)


# ── Ticker eşleştirme: isim → ticker.IS ─────────────────────────────
# BIST ticker'larının bilinen şirket isimleri. Eşleşme case-insensitive,
# kelime sınırlı. Genişletmek için scripts/build_ticker_aliases.py
TICKER_ALIASES = {
    "THYAO.IS":  ["THY", "Türk Hava Yolları", "Turkish Airlines", "THYAO"],
    "ASELS.IS":  ["ASELSAN", "Aselsan", "ASELS"],
    "GARAN.IS":  ["Garanti BBVA", "Garanti Bankası", "GARAN"],
    "AKBNK.IS":  ["Akbank", "AKBNK"],
    "YKBNK.IS":  ["Yapı Kredi", "Yapi Kredi", "YKBNK"],
    "ISCTR.IS":  ["İş Bankası", "Is Bankasi", "ISCTR"],
    "HALKB.IS":  ["Halkbank", "HALKB"],
    "VAKBN.IS":  ["VakıfBank", "Vakifbank", "VAKBN"],
    "KCHOL.IS":  ["Koç Holding", "Koc Holding", "KCHOL"],
    "SAHOL.IS":  ["Sabancı Holding", "Sabanci Holding", "SAHOL"],
    "TUPRS.IS":  ["TÜPRAŞ", "Tupras", "TUPRS"],
    "SISE.IS":   ["Şişecam", "Sisecam", "SISE"],
    "EREGL.IS":  ["Ereğli Demir", "Eregli Demir", "EREGL"],
    "BIMAS.IS":  ["BİM", "BIM Birleşik", "BIMAS"],
    "FROTO.IS":  ["Ford Otosan", "FROTO"],
    "TOASO.IS":  ["Tofaş", "Tofas", "TOASO"],
    "TCELL.IS":  ["Turkcell", "TCELL"],
    "TAVHL.IS":  ["TAV Havalimanları", "TAV Havalimanlari", "TAVHL"],
    "PGSUS.IS":  ["Pegasus", "PGSUS"],
    "EKGYO.IS":  ["Emlak Konut", "EKGYO"],
    "PETKM.IS":  ["Petkim", "PETKM"],
    "TTKOM.IS":  ["Türk Telekom", "Turk Telekom", "TTKOM"],
    "ARCLK.IS":  ["Arçelik", "Arcelik", "ARCLK"],
    "ENKAI.IS":  ["Enka İnşaat", "Enka Insaat", "ENKAI"],
    "TKFEN.IS":  ["Tekfen", "TKFEN"],
    "SASA.IS":   ["SASA Polyester", "SASA"],
    "GUBRF.IS":  ["Gübre Fabrikaları", "Gubre Fabrikalari", "GUBRF"],
    "DOHOL.IS":  ["Doğan Holding", "Dogan Holding", "DOHOL"],
    "MGROS.IS":  ["Migros", "MGROS"],
    "SOKM.IS":   ["ŞOK Marketler", "SOK Marketler", "SOKM"],
}


def _compile_pattern(aliases: list[str]) -> re.Pattern:
    """Whole-word, case-insensitive bir regex pattern."""
    escaped = [re.escape(a) for a in aliases]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE | re.UNICODE)


# Önbellekli compile
_PATTERNS: dict[str, re.Pattern] = {t: _compile_pattern(a) for t, a in TICKER_ALIASES.items()}


def match_tickers(text: str) -> list[str]:
    """Bir metin içinde geçen BIST ticker'larını döndürür."""
    if not text:
        return []
    return [t for t, pat in _PATTERNS.items() if pat.search(text)]


def _id_for_entry(url: str, title: str) -> str:
    """Dedup için stable ID — URL yoksa title hash."""
    base = (url or title or "").strip()
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def fetch_feed(name: str, url: str) -> list[dict]:
    """Tek bir RSS feed'i parse eder."""
    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        logger.warning(f"[{name}] feed parse fail: {e}")
        return []
    if parsed.bozo:
        logger.warning(f"[{name}] feed bozo: {parsed.bozo_exception}")
    items: list[dict] = []
    for e in parsed.entries:
        title = (e.get("title") or "").strip()
        summary = (e.get("summary") or e.get("description") or "").strip()
        url_ = e.get("link") or ""
        # Yayın tarihi
        published = e.get("published_parsed") or e.get("updated_parsed")
        if published:
            ts = datetime(*published[:6])
        else:
            ts = _utcnow()
        # Body için title + summary birleştir (RSS genelde tam metin vermez)
        body = (title + " — " + summary).strip(" —")
        tickers = match_tickers(body)
        if not tickers:
            continue
        for ticker in tickers:
            items.append({
                "id": _id_for_entry(url_, title) + "_" + ticker,
                "source": name,
                "url": url_,
                "title": title,
                "summary": summary[:1000],   # safety cap
                "published_at": ts,
                "ticker": ticker,
                "ingested_at": _utcnow(),
            })
    logger.info(f"[{name}] {len(items)} eşleşme bulundu")
    return items


def fetch_all() -> pd.DataFrame:
    """Tüm feed'leri sırayla çek, deduped DataFrame döndür ve diske yaz."""
    feeds = SETTINGS["ingestion"]["news_feeds"]
    all_items: list[dict] = []
    for f in feeds:
        all_items.extend(fetch_feed(f["name"], f["url"]))

    if not all_items:
        logger.info("Hiç eşleşen haber bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(all_items).drop_duplicates(subset=["id"]).sort_values("published_at")

    # Tarih bazında dosya — günlük partition
    today = _utcnow().date().isoformat()
    path = RAW_NEWS_DIR / f"news_{today}.parquet"

    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["id"])

    df.to_parquet(path, index=False)
    logger.info(f"News ingestion bitti — {len(df)} kayıt → {path.name}")
    return df


if __name__ == "__main__":
    fetch_all()
