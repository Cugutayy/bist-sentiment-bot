"""Triple Barrier Labeling — López de Prado, AFML §3.

Bir entry noktasından başlayarak şu üç bariyerden hangisi önce vurursa
ona göre etiket koyar:
- Üst bariyer (kar al, +pt): label = +1
- Alt bariyer (zarar kes, -sl): label = -1
- Zaman bariyeri (max holding): label = 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SETTINGS


def triple_barrier_labels(
    prices: pd.Series,
    pt_pct: float | None = None,
    sl_pct: float | None = None,
    max_holding: int | None = None,
) -> pd.Series:
    """Bir ticker'ın günlük close serisi → triple barrier label Series.

    Parameters
    ----------
    prices : pd.Series   index=date, value=close
    pt_pct : float       üst bariyer (örn. 0.025 = +%2.5)
    sl_pct : float       alt bariyer (örn. 0.015 = -%1.5)
    max_holding : int    en fazla N gün tut, sonra zaman bariyeri kapanır

    Returns
    -------
    pd.Series  aynı index, dtype int8, {-1, 0, +1}
    """
    cfg = SETTINGS["labels"]
    pt_pct      = pt_pct      if pt_pct      is not None else cfg["profit_target_pct"]
    sl_pct      = sl_pct      if sl_pct      is not None else cfg["stop_loss_pct"]
    max_holding = max_holding if max_holding is not None else cfg["max_holding_days"]

    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    arr = prices.to_numpy(dtype=float)

    for i in range(n):
        entry = arr[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(i + max_holding, n - 1)
        path = arr[i+1:end+1]
        if len(path) == 0:
            continue
        ret = (path - entry) / entry
        # İlk vurulan bariyer
        hit_up = np.where(ret >= pt_pct)[0]
        hit_dn = np.where(ret <= -sl_pct)[0]
        first_up = hit_up[0] if len(hit_up) else 10**9
        first_dn = hit_dn[0] if len(hit_dn) else 10**9
        if first_up < first_dn:
            labels[i] = +1
        elif first_dn < first_up:
            labels[i] = -1
        # else: zaman bariyeri → 0

    return pd.Series(labels, index=prices.index, name="label")


if __name__ == "__main__":
    # Mini test — sentetik fiyat serisinde label'lar mantıklı mı?
    rng = np.random.default_rng(42)
    n = 200
    prices = pd.Series(100 + rng.standard_normal(n).cumsum())
    labels = triple_barrier_labels(prices)
    print(labels.value_counts())
