# BIST Sentiment Bot

BIST 100 için günlük **top-N long shortlist** üreten araştırma pipeline'ı.
KAP açıklamaları + finans haberleri + Claude LLM tabanlı sentiment skorlaması
ile haftalık LightGBM modeli eğitir, walk-forward backtest ile doğrular,
her akşam telegram/email ile aday listesi gönderir.

> ⚠️ **Bu sistem otomatik emir göndermez.** Çıktısı: shortlist + "neden"
> kanıtı. Emir girişi insandadır.

## Tasarım Felsefesi

- **Honest backtest**: purged + embargoed walk-forward, realistik BIST cost
  modeli (komisyon + BSMV + spread), survivorship bias için point-in-time
  BIST 100 üyelik.
- **Modüler**: her aşama (ingestion → NLP → features → model → strategy)
  ayrı bir paket, unit-testli.
- **Reproducible**: fixed seeds, versiyonlu data (parquet timestamp'li),
  versiyonlu model (joblib + sha).
- **Cached LLM**: Claude API çağrıları content-hash ile cache'li; aynı
  haber tekrar skorlanmaz.

## Klasör Yapısı

```
config/         settings.yaml — tickers, eşikler, costs
data/raw/       immutable ham veri (parquet, append-only)
data/processed/ türetilmiş özellikler, sentiment_daily
ingestion/      veri toplayıcılar (KAP, news, price, reddit)
nlp/            sentiment Claude scoring + bot/spam filter
features/       feature engineering (her feature ayrı fonksiyon, lagged)
models/         LightGBM train/predict + triple barrier labeling
strategy/       shortlist üretimi
backtest/       walk-forward engine + realistik cost model
reports/        günlük email/telegram push
tests/          unit testler
scripts/        bir kerelik scriptler (geçmiş veri yükleme vs)
main.py         orchestrator (cron entry point)
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # ANTHROPIC_API_KEY doldur
```

## Kullanım

```bash
# Veri toplama (cron: her gün 18:30)
python -m ingestion.price_collector
python -m ingestion.news_collector
python -m ingestion.kap_collector       # Faz 1B, henüz aktif değil

# Sentiment skorlama (cron: her saat)
python -m nlp.sentiment_claude

# Günlük shortlist üretimi (cron: her gün 19:00)
python main.py predict --date today
```

## KAP Scraping (Faz 1B) — Setup

KAP, 2024 sonunda Next.js + Server Components mimarisine geçti. Standart
`requests` ile disclosure çekmek artık imkansız — headless Chrome
(Playwright) gerekli.

```bash
pip install playwright
python -m playwright install chromium    # bir kez, ~150MB
```

İlk reverse engineering adımı (KAP'ın gerçek XHR/Server Action
endpoint'lerini yakalamak için):

```bash
python scripts/explore_kap.py
```

Bu script bir Chrome instance açar, KAP bildirim sayfasını yükler ve
tüm network çağrılarını `data/raw/kap_debug/` altına yazar. Çıktıyı
inceledikten sonra `ingestion/kap_collector.py`'ye parse logic ekleyeceğiz.

## Yol Haritası

- **Faz 1A** — Repo + price + news + sentiment prototip + backtest iskeleti ✅
- **Faz 1B** — KAP Playwright iskelet ✅ · parse logic 🔄 · Reddit ⏳
- **Faz 2**  — Feature engineering + LightGBM training + walk-forward backtest
- **Faz 3**  — Shortlist üretimi + günlük email/telegram push
- **Faz 4**  — Canlı izleme (4-8 hafta), performans loglama

## Lisans

Özel proje — eğitim/araştırma amaçlı. Yatırım tavsiyesi değildir.
