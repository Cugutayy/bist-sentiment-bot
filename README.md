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
# 1) Veri toplama
python main.py ingest          # price + news, tek seferde

# 2) Backtest (model artifact üretir — predict için gerekli)
python main.py backtest        # 5y veride 6 fold walk-forward

# 3) Günlük shortlist (model artifact'ın bulunması gerekiyor)
python main.py predict         # bugün için top-N + neden

# 4) Sentiment skorlama (ANTHROPIC_API_KEY varsa)
python main.py score           # günün haberlerini Claude ile skorla

# Veri kalitesi denetimi
python scripts/audit_prices.py
```

### Cron Önerisi (Windows Task Scheduler veya Linux crontab)

```cron
# Hergün 18:30 — veri topla
30 18 * * 1-5  python /path/bist-sentiment-bot/main.py ingest

# Hergün 19:00 — sentiment skorla
0  19 * * 1-5  python /path/bist-sentiment-bot/main.py score

# Pazar 20:00 — modeli haftalık yeniden eğit (backtest model artifact üretir)
0  20 * * 0    python /path/bist-sentiment-bot/main.py backtest

# Hergün 19:15 — shortlist üret
15 19 * * 1-5  python /path/bist-sentiment-bot/main.py predict
```

## Bilinen yfinance / BIST Sorunları (UNUTMA)

Yahoo Finance, BIST verisi için **resmi/garanti** kaynak değil; unofficial
scraper. Şu sorunları sistemde defansif olarak ele aldık:

| Sorun | Belirti | Savunma |
|---|---|---|
| **Bedelsiz/split adjustment bug** | Tarihsel kapanışlarda anlamsız >%50 sıçrama. Özellikle BIMAS, EREGL, ASELS, FROTO. | `data_quality.check()` JUMP_PCT_ERROR=60% eşiği — log'a kırmızı uyarı. |
| **BIST tatil günleri NaN** | 19 Mayıs, 30 Ağustos, 29 Ekim, Ramazan/Kurban Bayramı | Modelde forward-fill veya gün-bazlı atlama. |
| **^XU100 vs XU100.IS** | Farklı scale dönerler (101.729 vs 13.163). Aynı endeks, farklı düzeltme. | Sistemde tek seçim: **XU100.IS** (`config/settings.yaml`). |
| **Stale data** | Yahoo bazen 1-2 iş günü geriden gelir. | `data_quality.check()` STALE_DAYS_WARN=3 — uyarı. |
| **Volume = 0** | Yahoo veri kaybı veya gerçekten işlem yok. | %5'ten fazla 0-volume → uyarı. |
| **chartPreviousClose mismatch** | regularMarketPrice ile arada uçurum varsa corporate action var. | Live fetch'te previousClose ham olarak log'lanır. |

`ingestion/data_quality.py` her fetch sonrası otomatik çalışır,
problemleri loguna düşer. Eşikler aynı dosyada — gerektikçe ayarla.

**Cross-source validation (Faz 2):** Yahoo'ya alternatif olarak
Investing.com / Bigpara / İş Yatırım scrape edip günlük kapanışları
karşılaştırma (>%1 sapma varsa kırmızı bayrak).

Audit komutu:
```bash
python scripts/audit_prices.py
```

## KAP Açıklamaları — Neden Doğrudan Scrape Etmiyoruz?

KAP, 2024 sonunda Next.js + React Server Components mimarisine geçti.
Standart HTTP istekleriyle disclosure çekmek artık imkansız — JS chunk
dosyalarında bile endpoint string'i yok (Server Action opaque payload).
Tek yol headless Chrome (Playwright) olurdu.

**Daha temiz çözüm:** KAP açıklamaları (bedelsiz hisse, temettü, kar
açıklaması, ihale alımı vb.) saniyeler içinde **AA, Hürriyet, Sabah,
BloombergHT, Yeni Şafak, Dünya, Habertürk** gibi haber kaynaklarına
mirror'lanıyor. Bu 7 RSS feed sentiment için yeterli sinyal sağlıyor
— hem teknik hem hukuki olarak temiz.

Feed'leri 6 ayda bir doğrulamak için (dropoff savunması):
```bash
# Faz 1B'de eklenecek:
python scripts/probe_feeds.py
```

## Yol Haritası

- **Faz 1A** ✅ Repo + price + 7-feed news + Claude sentiment + backtest iskeleti
- **Faz 1B** ✅ KAP → 7 RSS feed alternatifi (devlet sitesi scraping iptal)
- **Faz 2**  ✅ 14 feature + LightGBM + purged walk-forward + realistic cost model
- **Faz 3**  ✅ `strategy/shortlist.py` + `reports/console.py` + `main.py predict`
- **Faz 4**  🔄 Cron setup ✅ · sentiment integration ⏳ · Telegram push ⏳ · live performans tracking ⏳

## Mevcut Backtest Sonuçları (BIST 30, 5 yıl, 6 fold walk-forward)

| Metrik | Portföy | XU100 BH (aynı dönem) |
|---|---|---|
| Sharpe | **+0.17** | -0.82 |
| CAGR | +2.2% | +9.0% |
| Max DD | -37.6% | -11.3% |
| Hit rate | 48.8% | 49.4% |
| Turnover (günlük) | **2.4%** | — |
| n_days | 82 | 82 |

**Honest yorum:**
- Strateji XU100'ün negatif olduğu test günlerinde pozitif Sharpe üretti.
- CAGR düşük çünkü XU100 trend dönemleri test'in dışında kaldı.
- Max DD -37% yüksek — concentration riski (top-10 long, sektör cap yok).
- **Sentiment data yok** — fiyat-only baseline. Sentiment ekledikçe iyileşmesi beklenir.
- Turnover %94 → %2.4 (haftalık rebalance + sinyal EMA smoothing).

## Lisans

Özel proje — eğitim/araştırma amaçlı. Yatırım tavsiyesi değildir.
