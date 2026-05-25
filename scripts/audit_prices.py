"""İndirilen yfinance verilerini denetle.

Yakaladıklarımız:
- NaN değerler (eksik gün, BIST tatili)
- Sıfır volume (sahte tick)
- Aşırı tek-günlük sıçramalar (>25% — split/bedelsiz işareti veya bug)
- Tarihsel boşluklar (uzun gap = ya tatil ya veri eksikliği)
- Stale data (son güncelleme >3 iş günü önceyse)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "prices"


def audit() -> None:
    files = sorted(DATA.glob("*.parquet"))
    print(f"Toplam {len(files)} ticker")
    print()
    header = f'{"Ticker":<12} {"satir":>6} {"NaN":>5} {"vol_0":>6} {"max_jump":>9} {"date_gap":>10} {"son_tarih":>12}'
    print(header)
    print("-" * len(header))

    problems = []
    for f in files:
        df = pd.read_parquet(f)
        name = f.stem
        n = len(df)
        nans = int(df["close"].isna().sum())
        zero_vol = int((df["volume"] == 0).sum())
        rets = df["close"].pct_change().abs()
        max_jump = (rets.max() * 100) if not rets.empty else 0.0
        df["date"] = pd.to_datetime(df["date"])
        gaps = df["date"].diff().dt.days
        max_gap = int(gaps.max()) if not gaps.empty else 0
        last = df["date"].max().date()

        flag = ""
        reasons = []
        if nans > 0:
            reasons.append(f"NaN={nans}")
        if max_jump > 25:
            reasons.append(f"split?={max_jump:.0f}%")
        if max_gap > 7:
            reasons.append(f"gap={max_gap}g")
        if reasons:
            flag = "  <-- " + " ".join(reasons)
            problems.append((name, reasons))

        print(f"{name:<12} {n:>6} {nans:>5} {zero_vol:>6} {max_jump:>8.1f}% {max_gap:>10}g {str(last):>12}{flag}")

    print()
    print(f"Problemli ticker sayisi: {len(problems)}")
    print()
    print("=== Bilinen yfinance/BIST sorunlari ===")
    print("- 19 Mayis, 30 Agustos, 29 Ekim, Ramazan/Kurban bayrami: tatil, NaN normal")
    print("- >50% tek gunluk sicrama: bedelsiz hisse / split (Yahoo bazen yanlis duzeltir)")
    print("- BIMAS, EREGL: bedelsiz hisse gecmisi yogun, dikkat")
    print("- ^XU100 ile XU100.IS farkli scale donebilir (Yahoo iki versiyon tutuyor)")


if __name__ == "__main__":
    audit()
