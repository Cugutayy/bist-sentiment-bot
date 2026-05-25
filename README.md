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

## Web Dashboard (Streamlit)

Tüm backtest sonuçları, paper trade kayıtları, shortlist'i interaktif
görmek için:

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

7 sayfa: Özet · Backtest · Shortlist · Paper Trading · Sentiment ·
Veri Kalitesi · Konfig.

**Cloud deploy (ücretsiz)**: [share.streamlit.io](https://share.streamlit.io)
→ GitHub repo'yu bağla → `dashboard/app.py` seç → otomatik deploy.

Cloud dashboard sadece repo'daki "snapshot" veriyi gösterir. Sürekli
güncel veri için cron PC'de veya VPS'te çalışmalı (data klasörü
git'lenmez, sadece local).

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

## Sentiment Integration — Tek-Haber Dominasyonu Önleme

**Sorun:** Tek bir aşırı pozitif/negatif haber portföyü domine etmemeli;
piyasa zaten sentiment'i hızlıca fiyatlıyor (overreaction → mean-reversion).

**Çözüm — Literatür tabanlı 6 katmanlı koruma**

(Referanslar: [Kirtaç & Germano 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4706629),
[MANA-Net 2024](https://arxiv.org/html/2409.05698v1),
[Interpretable ML for Macro Alpha 2025](https://arxiv.org/pdf/2505.16136))

| Koruma | Nerede |
|---|---|
| **Winsorization** ±%95 quantile clip | `sentiment_aggregator._winsorize_series` |
| **Confidence weighting** (Claude'un güveni) | `weight = ... × confidence` |
| **Relevance weighting** (hisseyle ilgi) | `weight = ... × relevance` |
| **Source credibility** (KAP > Reuters > genel) | `settings.yaml` source_credibility tablosu |
| **Exponential decay** half-life 5 gün | `_decay_weight(age, half_life)` |
| **Surprise feature** (sentiment − rolling_30d) | `feature_engineering._add_sentiment_features` |

**Feature çıktısı** (model girişine 8 sentiment feature):
- `sent_w_3d`, `sent_w_7d`, `sent_w_14d` — ağırlıklı rolling mean (multi-horizon)
- `sent_surprise` = today − rolling_30d (mean reversion baseline'dan sapma)
- `sent_momentum` = EMA(7) − EMA(30) (trend yakalama)
- `sent_news_count_7d` — son hafta haber sayısı
- `sent_news_surge` = bugün / 30d ortalama (anormal aktivite)
- `sent_std_7d` — sentiment uzlaşmazlığı (yüksek = belirsiz)

**Graceful fallback**: sentiment data yoksa tüm sent_* feature'lar 0,
model fiyat-only baseline'a düşer (her zaman çalışır).

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
