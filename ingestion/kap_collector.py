"""KAP (Kamuyu Aydınlatma Platformu) açıklama scraper'ı.

Neden Playwright?
-----------------
KAP 2024 sonunda yeni Next.js tabanlı siteye geçti. Tüm disclosure
endpoint'leri React Server Components (RSC) Server Action'ları arkasında
— `requests` kütüphanesiyle yakalanamayan opaque POST payload'ları.
JS bundle'larında bile endpoint string'i çıkmıyor.

Çözüm: gerçek headless Chrome (Playwright) ile sayfayı render etmek
ve `page.on("response", ...)` ile XHR/Server Action yanıtlarını
yakalamak. Tek pratik yol.

Kurulum
-------
    pip install playwright
    python -m playwright install chromium     # ~150MB, bir kez

Kullanım
--------
    # Hızlı test:
    python scripts/explore_kap.py

    # Bir tickerın son 30 günlük açıklamaları:
    python -m ingestion.kap_collector --ticker ASELS --days 30
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config import ROOT

RAW_KAP_DIR = ROOT / "data" / "raw" / "kap"
RAW_KAP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class KapDisclosure:
    """KAP'tan çekilen tek bir açıklama."""
    disclosure_id: str           # KAP'ın kendi index ID'si
    ticker: str                  # ASELS, THYAO, ...
    company_name: str
    disclosure_date: datetime    # bildirim tarihi+saati
    subject: str                 # konu (örn. "Özel Durum Açıklaması")
    summary: str                 # başlık / kısa metin
    url: str                     # detay sayfa
    raw: dict[str, Any] = field(default_factory=dict)   # ham JSON


def _import_playwright():
    """Lazy import — Playwright kurulu olmayan ortamda crash etme."""
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright kurulu değil. Kur:\n"
            "    pip install playwright\n"
            "    python -m playwright install chromium"
        ) from e


def fetch_disclosures(
    ticker: str | None = None,
    days_back: int = 30,
    headless: bool = True,
    debug_dump_responses: bool = False,
) -> list[KapDisclosure]:
    """KAP bildirim-sorgu sayfasından açıklamalar çek.

    Parameters
    ----------
    ticker : str, optional
        Tek ticker filtresi (örn. "ASELS"). None ise tüm BIST.
    days_back : int
        Bugünden kaç gün geriye bakılsın.
    headless : bool
        False ise gerçek Chrome açılır (debug için).
    debug_dump_responses : bool
        True ise tüm XHR/Server Action yanıtları /tmp/kap_responses/'a yazılır.
        Endpoint reverse engineering için.

    Returns
    -------
    list[KapDisclosure]
    """
    sync_playwright = _import_playwright()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    captured_responses: list[dict] = []
    disclosures: list[KapDisclosure] = []

    dump_dir = ROOT / "data" / "raw" / "kap_debug"
    if debug_dump_responses:
        dump_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/130.0.0.0 Safari/537.36",
            locale="tr-TR",
        )
        page = context.new_page()

        # Tüm XHR/fetch yanıtlarını yakala
        def on_response(resp):
            url = resp.url
            ct = (resp.headers.get("content-type") or "").lower()
            if "kap.org.tr" not in url:
                return
            # İlgi alanı: JSON ya da text payload, server actions
            is_xhr = resp.request.resource_type in ("xhr", "fetch")
            is_json = "json" in ct or "rsc" in ct
            if not (is_xhr or is_json):
                return
            try:
                body = resp.text()
            except Exception:
                return
            captured_responses.append({
                "url": url,
                "method": resp.request.method,
                "status": resp.status,
                "content_type": ct,
                "size": len(body),
                "body_preview": body[:500],
            })
            if debug_dump_responses:
                safe = url.replace("https://www.kap.org.tr/", "").replace("/", "_")[:80]
                ts = datetime.now().strftime("%H%M%S")
                (dump_dir / f"{ts}_{safe}.txt").write_text(
                    f"URL: {url}\nMETHOD: {resp.request.method}\n"
                    f"STATUS: {resp.status}\nCT: {ct}\n\n{body}",
                    encoding="utf-8",
                )

        page.on("response", on_response)

        # Sayfaya git — bildirim sorgu (ana liste)
        logger.info(f"KAP bildirim-sorgu açılıyor (last {days_back} gün)...")
        page.goto("https://www.kap.org.tr/tr/bildirim-sorgu", wait_until="networkidle", timeout=30_000)

        # Sayfa render olduktan sonra ilk DOM elemanlarına bak
        # (XHR'lar bu sırada gerçekleşir; on_response yakalar)
        page.wait_for_timeout(3_000)

        # TODO: ticker filtresi ve tarih aralığı için form etkileşimi
        # — bu Faz 1B sonraki commit'te, explore_kap.py log'undan sonra
        # implement edilecek.

        browser.close()

    logger.info(f"Yakalanan XHR/Server Action yanıtları: {len(captured_responses)}")
    for r in captured_responses[:5]:
        logger.info(f"  {r['method']} {r['status']} | {r['size']:>6}b | {r['url'][:80]}")

    if debug_dump_responses:
        logger.info(f"Tüm yanıtlar yazıldı → {dump_dir}")

    # Faz 1B 2. commit'te: captured_responses'tan disclosure'ları parse et
    return disclosures


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--show-browser", action="store_true", help="Chrome'u görünür aç")
    p.add_argument("--dump", action="store_true",
                   help="Tüm yanıtları data/raw/kap_debug/'a yaz (endpoint reverse engineering)")
    args = p.parse_args()

    fetch_disclosures(
        ticker=args.ticker,
        days_back=args.days,
        headless=not args.show_browser,
        debug_dump_responses=args.dump,
    )
