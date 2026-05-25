"""Terminal raporlama — günlük shortlist için temiz tablo çıktısı."""
from __future__ import annotations

from datetime import datetime

from strategy.shortlist import ShortlistItem


def print_shortlist(items: list[ShortlistItem], as_of: str | None = None) -> None:
    """Shortlist'i konsola yazdır — renkli olmadan, sadece tablo."""
    date_str = as_of or datetime.now().strftime("%Y-%m-%d")
    print()
    print("=" * 95)
    print(f"BIST SENTIMENT BOT — Günlük Shortlist ({date_str})")
    print("=" * 95)
    print(f"{'#':>2} {'Ticker':<10} {'Skor':>6} {'Fiyat':>10} {'5g_ret':>8} {'Mom60':>7} {'RSI':>5}  Neden")
    print("-" * 95)
    for it in items:
        print(
            f"{it.rank:>2} {it.ticker:<10} "
            f"{it.score:>6.3f} {it.last_close:>10,.2f} "
            f"{it.last_return_5d_pct:>+7.2f}% "
            f"{it.momentum_60d_pct:>+6.1f}% "
            f"{it.rsi_14:>5.1f}  {it.rationale}"
        )
    print()
    print("Not: skor 0..1, model predict_proba (triple-barrier +1 hit olasılığı).")
    print("     Neden = en yüksek importance × feature değeri sahip 3 sinyal.")
    print("     YATIRIM TAVSİYESİ DEĞİLDİR. Manuel doğrula, kendi araştırmanı yap.")
    print()
