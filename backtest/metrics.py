"""Sharpe, Max DD, CAGR, hit rate, monthly breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_FACTOR = 252  # işlem günü


def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    sd = excess.std()
    if sd == 0 or pd.isna(sd):
        return float("nan")
    return float(np.sqrt(ANNUAL_FACTOR) * excess.mean() / sd)


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    eq = (1 + returns).cumprod()
    dd = (eq / eq.cummax()) - 1
    return float(dd.min())


def cagr(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    years = len(returns) / ANNUAL_FACTOR
    if years <= 0:
        return float("nan")
    return float(total ** (1 / years) - 1)


def hit_rate(returns: pd.Series) -> float:
    n = (returns != 0).sum()
    if n == 0:
        return float("nan")
    return float((returns > 0).sum() / n)


def turnover(weights: pd.DataFrame) -> float:
    """Günlük ortalama turnover (0-1 arası)."""
    if weights.empty:
        return float("nan")
    diff = weights.diff().abs().sum(axis=1)
    return float(diff.mean())


def summary(returns: pd.Series, weights: pd.DataFrame | None = None, rf_annual: float = 0.40) -> dict:
    """Tek bir dict — tüm metrikler. rf_annual TR risk-free (~politika faizi)."""
    rf_daily = rf_annual / ANNUAL_FACTOR
    out = {
        "n_days":   int(len(returns)),
        "sharpe":   round(sharpe(returns, rf_daily), 3),
        "cagr":     round(cagr(returns), 4),
        "max_dd":   round(max_drawdown(returns), 4),
        "hit_rate": round(hit_rate(returns), 3),
        "mean_d":   round(float(returns.mean()), 5),
        "std_d":    round(float(returns.std()), 5),
    }
    if weights is not None and not weights.empty:
        out["turnover_d"] = round(turnover(weights), 3)
    return out
