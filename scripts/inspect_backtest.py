"""Backtest sonuçlarını detaylı incele.

Backtest çalıştıktan sonra data/processed/backtests/<run_id>/ altında
artifact'lar üretilir. Bu script en son run'ı yükleyip:
- Aylık return breakdown
- Top kazançlı / kayıplı günler
- Per-ticker katkı (hangi hisse PnL'in ne kadarını üretti)
- Equity curve ASCII grafik
- Fold-by-fold Sharpe

Kullanım:
    python scripts/inspect_backtest.py             # en son run
    python scripts/inspect_backtest.py --run-id 20260525_123456_xxx
    python scripts/inspect_backtest.py --plot       # matplotlib grafik (.png)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT

BT_DIR = ROOT / "data" / "processed" / "backtests"


def _find_latest_run() -> Path | None:
    runs = sorted(BT_DIR.glob("*/"), reverse=True)
    return runs[0] if runs else None


def _ascii_equity_curve(equity: pd.Series, width: int = 60, height: int = 12) -> str:
    """Equity curve'i ASCII art olarak çiz."""
    if len(equity) < 2:
        return "(yetersiz veri)"
    eq = equity.values
    e_min, e_max = float(eq.min()), float(eq.max())
    rng = e_max - e_min or 1e-9
    cols = min(len(eq), width)
    step = len(eq) / cols
    sampled = [eq[int(i * step)] for i in range(cols)]
    chart = [[" "] * cols for _ in range(height)]
    for x, v in enumerate(sampled):
        y = int((1 - (v - e_min) / rng) * (height - 1))
        y = max(0, min(height - 1, y))
        chart[y][x] = "●"
    lines = []
    for y, row in enumerate(chart):
        v_at_y = e_max - (y / (height - 1)) * rng if height > 1 else e_max
        label = f"{v_at_y:>6.3f} |"
        lines.append(label + "".join(row))
    lines.append(" " * 8 + "-" * cols)
    lines.append(f"        {equity.index[0].date()}{' ' * (cols - 22)}{equity.index[-1].date()}")
    return "\n".join(lines)


def inspect(run_id: str | None = None, do_plot: bool = False) -> None:
    if run_id:
        run_dir = BT_DIR / run_id
    else:
        run_dir = _find_latest_run()
    if run_dir is None or not run_dir.exists():
        print("Hiç backtest run bulunamadı. Önce: python main.py backtest")
        return

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(run_dir / "daily.parquet").set_index("date")
    weights = pd.read_parquet(run_dir / "weights.parquet").set_index("date")

    print("=" * 78)
    print(f"BACKTEST INCELEMESI — {run_dir.name}")
    print("=" * 78)
    print()
    print(f"Run ID:           {summary['run_id']}")
    print(f"Fold sayısı:      {summary['n_folds']}")
    print(f"İşlem günü:       {summary['n_days']}")
    print()

    pm = summary["portfolio_metrics"]
    bm = summary["benchmark_metrics"]
    print(f"{'Metrik':<15} {'Portföy':>12} {'XU100 BH':>12}")
    print("-" * 42)
    for k in ("sharpe", "cagr", "max_dd", "hit_rate", "mean_d", "std_d"):
        print(f"{k:<15} {pm.get(k, '—'):>12} {bm.get(k, '—'):>12}")
    if "turnover_d" in pm:
        print(f"{'turnover_d':<15} {pm['turnover_d']:>12}")
    print()

    # Equity curve (ASCII)
    print("EQUITY CURVE (Portföy):")
    print(_ascii_equity_curve(daily["portfolio_equity"]))
    print()
    print("EQUITY CURVE (XU100 BH karşılaştırma):")
    print(_ascii_equity_curve(daily["benchmark_equity"]))
    print()

    # Aylık return
    print("AYLIK RETURN'LER (portföy):")
    monthly = daily["portfolio_return"].groupby(pd.Grouper(freq="ME")).apply(
        lambda s: (1 + s).prod() - 1
    )
    for date, ret in monthly.items():
        bar = "+" * int(max(0, ret) * 100) if ret >= 0 else "-" * int(abs(ret) * 100)
        bar = bar[:30]
        print(f"  {date.strftime('%Y-%m')}: {ret*100:+7.2f}%  {bar}")
    print()

    # Top + worst günler
    print("EN İYİ 5 GÜN:")
    top = daily["portfolio_return"].nlargest(5)
    for d, r in top.items():
        active = weights.loc[d][weights.loc[d] > 0].index.tolist() if d in weights.index else []
        print(f"  {d.date()}: {r*100:+6.2f}%  ({len(active)} pozisyon: {', '.join(active[:5])}{'...' if len(active) > 5 else ''})")
    print()
    print("EN KÖTÜ 5 GÜN:")
    worst = daily["portfolio_return"].nsmallest(5)
    for d, r in worst.items():
        active = weights.loc[d][weights.loc[d] > 0].index.tolist() if d in weights.index else []
        print(f"  {d.date()}: {r*100:+6.2f}%  ({len(active)} pozisyon: {', '.join(active[:5])}{'...' if len(active) > 5 else ''})")
    print()

    # Per-ticker katkı (yaklaşık)
    print("PER-TICKER PORTFÖYE EN ÇOK GİREN TOP 10:")
    ticker_active = (weights > 0).sum(axis=0).sort_values(ascending=False).head(10)
    for t, n in ticker_active.items():
        avg_w = weights[t][weights[t] > 0].mean() * 100
        print(f"  {t:<12} {n:>4} gün portföyde, ort. ağırlık %{avg_w:.1f}")
    print()

    # Fold breakdown
    print("FOLD-BY-FOLD:")
    print(f"  {'#':>3} {'Test dönemi':<35} {'n_train':>8} {'pos_rate':>9}")
    for fm in summary["fold_metrics"]:
        print(f"  {fm['fold']:>3} {fm['test_period']:<35} {fm['train_n']:>8} {fm['positive_rate_train']:>9.3f}")
    print()

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(daily.index, daily["portfolio_equity"], label="Portföy")
            ax1.plot(daily.index, daily["benchmark_equity"], label="XU100 BH", alpha=0.7)
            ax1.legend(); ax1.set_title("Equity Curve"); ax1.grid(True, alpha=0.3)
            monthly.plot.bar(ax=ax2, color=monthly.apply(lambda x: "g" if x > 0 else "r"))
            ax2.set_title("Aylık Return"); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = run_dir / "equity_curve.png"
            plt.savefig(plot_path, dpi=100)
            print(f"Grafik kaydedildi: {plot_path}")
        except ImportError:
            print("Matplotlib yok, --plot atlandı.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    inspect(run_id=args.run_id, do_plot=args.plot)
