"""Survivorship bias detection ve disclosure.

PROBLEM: Bugünkü BIST 30/100 üye listesini geçmişe uygulayarak yapılan
backtest GERÇEKDEN DAHA İYİ sonuç verir, çünkü:
- Bugün listede olan hisseler "hayatta kalanlar"dır
- Tarihsel olarak listede olup sonra "öldürülmüş" (delisting / iflas /
  füzyon) hisseler dataset'te YOKTUR
- Bu "hayatta kalan" bias backtest CAGR'ı %3-10 puan şişirebilir

Standart düzeltme: point-in-time membership (geçmişte hangi tarihte
hangi hisseler BIST 30/100'deydi) — özel veri kaynağı gerektirir
(Bloomberg, Refinitiv, BIST resmi tarih arşivi).

Bizim mevcut çözümümüz: BIAS YOK SAYMAK YERINE AÇIKÇA DEKLARE ETMEK.
Her backtest çıktısında uyarı, README'de prominent disclaimer.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from config import ROOT


def disclose_survivorship_bias() -> str:
    """Backtest report'una eklenecek standart disclosure metni."""
    return (
        "⚠️  SURVIVORSHIP BIAS UYARISI\n"
        "Backtest, BUGÜNKÜ BIST evren listesini 5 yıl geriye uygulayarak\n"
        "yapılmaktadır. Bu süre içinde 'öldürülmüş' (delisting / iflas)\n"
        "şirketler dataset'te YOKTUR. Bu, gerçek tarihsel performanstan\n"
        "%3-10 daha iyimser sonuç verebilir.\n"
        "\n"
        "Düzeltme için gerekli (Faz 5+): point-in-time BIST üye tarihsel\n"
        "üyelik veritabanı. Mevcut yfinance ücretsiz tier bunu sağlamıyor.\n"
        "\n"
        "Yorum: Backtest Sharpe'ını bu yüzden %15-25 indirimle değerlendir.\n"
    )


def estimate_bias_magnitude() -> dict:
    """5 yıllık BIST 30 listesinde değişim oranı tahmini.

    Akademik çalışmalar (Brown et al. 1992, Carhart 1997):
    - ABD piyasalarında survivorship bias yıllık ~%1-2 outperformance
    - Gelişmekte olan piyasalarda (BIST) ~%2-3 daha yüksek olabilir
    - 5 yılda kümülatif: ~%5-15 fazladan CAGR
    """
    return {
        "estimated_annual_bias_pct": "1-3",
        "cumulative_bias_5yr_pct":   "5-15",
        "confidence":                "düşük (resmi membership data olmadan)",
        "recommendation":            "Backtest Sharpe'ı 0.85 ile çarpıp realistik tahmin yap",
    }


if __name__ == "__main__":
    print(disclose_survivorship_bias())
    print("Tahmini bias şiddeti:", estimate_bias_magnitude())
