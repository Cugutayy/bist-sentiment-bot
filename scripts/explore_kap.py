"""KAP Reverse Engineering Script — bir kerelik çalıştır, log üret.

Bu script:
1) Playwright ile KAP bildirim-sorgu sayfasını gerçek Chrome'da açar
2) Sayfanın yaptığı TÜM XHR/Server Action çağrılarını yakalar
3) /tmp/kap_responses/'a hepsini yazar
4) Konsola özet basar

Çıktıyı bana gönderirsin → KAP'ın gerçek endpoint URL'lerini görürüz,
hangi POST gövdesi nasıl format → bir sonraki adımda parse logic yazarız.

Kullanım:
    pip install playwright
    python -m playwright install chromium
    python scripts/explore_kap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root'u path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.kap_collector import fetch_disclosures


if __name__ == "__main__":
    print("=" * 60)
    print("KAP Explore — XHR/Server Action yakalama")
    print("=" * 60)
    print()
    print("Bir Chrome instance açılacak, KAP bildirim-sorgu sayfasını")
    print("yükleyecek ve yapılan tüm network çağrılarını loglayacak.")
    print()
    print("Çıktı yazıldı:")
    print("  - data/raw/kap_debug/*.txt (her yanıt ayrı dosya)")
    print()
    print("Komut tamamlandığında konsoldaki özeti + ilk birkaç")
    print("yanıt dosyasının içeriğini bana gönder.")
    print()

    fetch_disclosures(
        ticker=None,
        days_back=7,
        headless=False,         # Browser görünür — emin olalım sayfa yükleniyor
        debug_dump_responses=True,
    )

    print()
    print("Bitti. data/raw/kap_debug/ klasöründeki dosyaları paylaş.")
