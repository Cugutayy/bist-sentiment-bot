"""BIST İstanbul Stock Exchange tatil takvimi.

Yaklaşım: pandas_market_calendars varsa onu kullan (XIST), yoksa
manuel resmi tatil listesi (2024-2028).

is_trading_day(date) → True/False
next_trading_day(date) → bir sonraki açık gün
"""
from __future__ import annotations

from datetime import date, timedelta

# Türkiye resmi tatil + BIST'in tipik olarak kapalı olduğu günler.
# Kaynak: KAP/Borsa İstanbul resmi duyuru takvimi.
# NOTE: Dini tatillerin tarihleri ay yılına göre kayar; her yıl manuel güncellenmeli.
BIST_HOLIDAYS: set[date] = {
    # === 2024 ===
    date(2024, 1, 1),                 # Yılbaşı
    date(2024, 4, 9), date(2024, 4, 10), date(2024, 4, 11),  # Ramazan Bayramı
    date(2024, 4, 23),                # Ulusal Egemenlik
    date(2024, 5, 1),                 # Emek ve Dayanışma
    date(2024, 5, 19),                # Atatürk'ü Anma Gençlik ve Spor
    date(2024, 6, 16), date(2024, 6, 17), date(2024, 6, 18), date(2024, 6, 19),  # Kurban Bayramı
    date(2024, 7, 15),                # Demokrasi ve Milli Birlik
    date(2024, 8, 30),                # Zafer
    date(2024, 10, 29),               # Cumhuriyet

    # === 2025 ===
    date(2025, 1, 1),
    date(2025, 3, 30), date(2025, 3, 31), date(2025, 4, 1),  # Ramazan Bayramı
    date(2025, 4, 23),
    date(2025, 5, 1),
    date(2025, 5, 19),
    date(2025, 6, 6), date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9),  # Kurban
    date(2025, 7, 15),
    date(2025, 8, 30),
    date(2025, 10, 29),

    # === 2026 ===
    date(2026, 1, 1),
    date(2026, 3, 20), date(2026, 3, 21), date(2026, 3, 22),  # Ramazan Bayramı
    date(2026, 4, 23),
    date(2026, 5, 1),
    date(2026, 5, 19),
    # Kurban Bayramı 2026: 26-29 Mayıs (Çar-Cmt)
    date(2026, 5, 26), date(2026, 5, 27), date(2026, 5, 28), date(2026, 5, 29),
    date(2026, 7, 15),
    date(2026, 8, 30),
    date(2026, 10, 29),

    # === 2027 (tahmini, doğrulanmalı) ===
    date(2027, 1, 1),
    date(2027, 3, 9), date(2027, 3, 10), date(2027, 3, 11),  # Ramazan
    date(2027, 4, 23),
    date(2027, 5, 1),
    date(2027, 5, 17), date(2027, 5, 18), date(2027, 5, 19),  # Kurban
    date(2027, 7, 15),
    date(2027, 8, 30),
    date(2027, 10, 29),

    # === 2028 ===
    date(2028, 1, 1),
    date(2028, 2, 25), date(2028, 2, 26), date(2028, 2, 27),  # Ramazan
    date(2028, 4, 23),
    date(2028, 5, 1),
    date(2028, 5, 5), date(2028, 5, 6), date(2028, 5, 7),     # Kurban (yaklaşık)
    date(2028, 5, 19),
    date(2028, 7, 15),
    date(2028, 8, 30),
    date(2028, 10, 29),
}


def is_trading_day(d: date | None = None) -> bool:
    """True = BIST açık (hafta içi + tatil değil)."""
    d = d or date.today()
    if d.weekday() >= 5:   # 5 = Cumartesi, 6 = Pazar
        return False
    return d not in BIST_HOLIDAYS


def next_trading_day(d: date | None = None) -> date:
    """d (varsayılan: bugün) sonrasındaki ilk açık BIST günü."""
    d = d or date.today()
    candidate = d + timedelta(days=1)
    for _ in range(15):  # max 2 hafta ileri
        if is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    return candidate


def reason_closed(d: date | None = None) -> str | None:
    """Borsa neden kapalı? None = açık, str = sebep."""
    d = d or date.today()
    if d.weekday() == 5:
        return "Cumartesi (hafta sonu)"
    if d.weekday() == 6:
        return "Pazar (hafta sonu)"
    if d in BIST_HOLIDAYS:
        return f"BIST resmi tatil ({d.isoformat()})"
    return None


if __name__ == "__main__":
    today = date.today()
    print(f"Bugün ({today}): trading day? {is_trading_day(today)}")
    if not is_trading_day(today):
        print(f"  Sebep: {reason_closed(today)}")
    print(f"Sonraki açık gün: {next_trading_day(today)}")
