"""yfinance veri kalitesi savunması.

Yahoo Finance BIST verisi için BİLİNEN sorunlar:
1) Bedelsiz hisse (bonus issue) / split düzeltmesi bazen yanlış yapılır.
   Özellikle BIMAS, EREGL, ASELS gibi sık bedelsiz dağıtan şirketlerde
   tarihsel kapanışlar arasında anlamsız >%50 sıçrama görünebilir.
2) BIST resmi tatil günlerinde (19 Mayıs, 30 Ağustos, 29 Ekim,
   bayramlar) bazen NaN, bazen önceki gün carry-forward döner —
   tutarsız.
3) ^XU100 (101.729) ve XU100.IS (13.163) farklı scale döner.
   Aynı index, farklı düzeltme. Sistem içinde TEK BİRİNİ seç ve sıkı sıkı.
4) Stale data — Yahoo bazen 1-2 iş günü gecikir, son satır eski olabilir.
5) Volume = 0 günler — Yahoo veri eksikliği ya da gerçekten işlem yok.
6) chartPreviousClose ile regularMarketPrice arasında uçurum varsa
   o gün corporate action var demektir (temettü, bedelsiz, vb.)

Bu modül her fetch sonrası çağrılır, problemli durumları loglar.
Otomatik düzeltme YOK — sadece görünür uyarı; modelin temizleme
katmanı bu uyarıları görerek karar verir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

Severity = Literal["info", "warn", "error"]


@dataclass
class QualityIssue:
    ticker: str
    kind: str            # 'nan', 'jump', 'gap', 'stale', 'zero_volume', 'scale'
    severity: Severity
    detail: str


# ─── Eşikler ───────────────────────────────────────────────────────
NAN_THRESHOLD = 5                # 5'ten fazla NaN: uyarı
JUMP_PCT_WARN = 25               # tek gün %25+ değişim: bedelsiz/split şüphesi
JUMP_PCT_ERROR = 60              # %60+ değişim: kesinlikle düzeltme bug'ı
GAP_DAYS_WARN = 7                # 1 haftadan uzun boşluk: tatil veya eksik
GAP_DAYS_ERROR = 14              # 2+ hafta gap: ciddi eksik
STALE_DAYS_WARN = 3              # son veri 3+ iş günü önceyse stale
ZERO_VOL_PCT_WARN = 0.05         # toplam günlerin %5'inden fazlası 0-volume


def check(ticker: str, df: pd.DataFrame) -> list[QualityIssue]:
    """Bir ticker'ın price DataFrame'i için kalite kontrol listesi döndür."""
    issues: list[QualityIssue] = []
    if df.empty:
        issues.append(QualityIssue(ticker, "empty", "error", "DataFrame boş"))
        return issues

    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1) NaN kontrolü
    nans = int(df["close"].isna().sum())
    if nans > NAN_THRESHOLD:
        issues.append(QualityIssue(ticker, "nan", "warn",
                                   f"{nans} NaN değer (>{NAN_THRESHOLD} eşik)"))

    # 2) Aşırı tek-günlük sıçrama
    rets = df["close"].pct_change().abs()
    max_jump_pct = float(rets.max() * 100) if not rets.empty else 0.0
    if max_jump_pct >= JUMP_PCT_ERROR:
        issues.append(QualityIssue(ticker, "jump", "error",
                                   f"Tek gün {max_jump_pct:.1f}% sıçrama — split/bedelsiz düzeltme bug'ı"))
    elif max_jump_pct >= JUMP_PCT_WARN:
        issues.append(QualityIssue(ticker, "jump", "warn",
                                   f"Tek gün {max_jump_pct:.1f}% sıçrama — corporate action?"))

    # 3) Tarihsel boşluklar
    gaps = df["date"].diff().dt.days
    max_gap = int(gaps.max()) if not gaps.empty else 0
    if max_gap >= GAP_DAYS_ERROR:
        issues.append(QualityIssue(ticker, "gap", "error",
                                   f"{max_gap} günlük veri boşluğu"))
    elif max_gap > GAP_DAYS_WARN:
        issues.append(QualityIssue(ticker, "gap", "warn",
                                   f"{max_gap} günlük boşluk (hafta sonu + tatil normal)"))

    # 4) Stale check
    last_date = df["date"].max()
    business_days_ago = pd.bdate_range(last_date, pd.Timestamp.now(tz="UTC").tz_localize(None)).size - 1
    if business_days_ago > STALE_DAYS_WARN:
        issues.append(QualityIssue(ticker, "stale", "warn",
                                   f"Son veri {business_days_ago} iş günü önce ({last_date.date()})"))

    # 5) Yüksek sıfır-volume oranı
    zero_vol_pct = float((df["volume"] == 0).mean())
    if zero_vol_pct > ZERO_VOL_PCT_WARN:
        n0 = int((df["volume"] == 0).sum())
        issues.append(QualityIssue(ticker, "zero_volume", "warn",
                                   f"{n0} gün ({zero_vol_pct*100:.1f}%) sıfır volume"))

    return issues


def log_issues(issues: list[QualityIssue]) -> None:
    """Issue listesini loguna düş — severity'ye göre."""
    for i in issues:
        msg = f"[QC {i.ticker}] {i.kind} → {i.detail}"
        if i.severity == "error":
            logger.error(msg)
        elif i.severity == "warn":
            logger.warning(msg)
        else:
            logger.info(msg)


def check_and_log(ticker: str, df: pd.DataFrame) -> list[QualityIssue]:
    """check() + log_issues() kombine."""
    issues = check(ticker, df)
    log_issues(issues)
    return issues
