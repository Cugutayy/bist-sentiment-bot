"""Backtest + shortlist + sentiment artifact'larını GitHub'a push.

Streamlit Cloud dashboard'unun güncel veri görmesi için.
Raw price data git'lenmez (büyük) — sadece küçük artifact'lar.

Kullanım:
    python scripts/sync_artifacts.py          # her zaman çalıştır
    # Veya cron: günde 1 kez backtest sonrası

Ne yapar:
    1. data/processed/backtests/*/summary.json + daily.parquet + weights.parquet
    2. data/processed/paper_trades.parquet (güncel paper log)
    3. data/processed/sentiment_daily.parquet (sentiment agregat)
    4. data/processed/shortlist_*.parquet (en son N)
    Diff varsa: git add + commit + push.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

ARTIFACT_PATHS = [
    "data/processed/backtests",
    "data/processed/paper_trades.parquet",
    "data/processed/sentiment_daily.parquet",
]


def run(cmd: list[str], check: bool = True) -> tuple[int, str]:
    """Komutu repo dizininde çalıştır, exit + stdout döndür."""
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}\n{r.stderr}")
    return r.returncode, (r.stdout + r.stderr)


def main() -> int:
    print("=== sync_artifacts.py ===")

    # Sadece var olan path'leri ekle
    to_add = []
    for p in ARTIFACT_PATHS:
        full = ROOT / p
        if full.exists():
            to_add.append(p)
    # Shortlist dosyaları (en son 7)
    shortlists = sorted(PROCESSED.glob("shortlist_*.parquet"), reverse=True)[:7]
    for s in shortlists:
        to_add.append(str(s.relative_to(ROOT)).replace("\\", "/"))

    if not to_add:
        print("Hiç artifact yok, atlanıyor.")
        return 0

    # Git add
    rc, out = run(["git", "add", *to_add], check=False)
    if rc != 0:
        print(out)
        return rc

    # Değişiklik var mı?
    rc, out = run(["git", "diff", "--staged", "--name-only"], check=False)
    staged = [l.strip() for l in out.splitlines() if l.strip()]
    if not staged:
        print("Hiç değişiklik yok, commit yapılmadı.")
        return 0
    print(f"{len(staged)} dosya stage'lendi:")
    for f in staged[:10]:
        print(f"  - {f}")
    if len(staged) > 10:
        print(f"  ... +{len(staged)-10} daha")

    # Commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"chore(data): sync artifacts {ts}\n\nAutomated by sync_artifacts.py"
    rc, out = run(["git", "commit", "-m", msg], check=False)
    if rc != 0:
        print("Commit fail:", out)
        return rc
    print(f"Commit edildi.")

    # Push
    rc, out = run(["git", "push"], check=False)
    if rc != 0:
        print("Push fail:", out)
        return rc
    print("Push OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
