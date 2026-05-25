"""Triple barrier label doğruluğunu deterministik örneklerle doğrula."""
import numpy as np
import pandas as pd

from models.triple_barrier import triple_barrier_labels


def test_upper_barrier_hit_first():
    # 100 → 103 (gün 1, +%3, pt=%2.5 üstünde) → label = +1
    prices = pd.Series([100, 103, 95])
    labels = triple_barrier_labels(prices, pt_pct=0.025, sl_pct=0.015, max_holding=5)
    assert labels.iloc[0] == 1


def test_lower_barrier_hit_first():
    # 100 → 98 (gün 1, -%2 <= -%1.5) → label = -1
    prices = pd.Series([100, 98, 105])
    labels = triple_barrier_labels(prices, pt_pct=0.025, sl_pct=0.015, max_holding=5)
    assert labels.iloc[0] == -1


def test_time_barrier_neutral():
    # 100 → 100.5 (+%0.5, hiç bariyere değmez) max=3 → label = 0
    prices = pd.Series([100, 100.5, 100.2, 100.3])
    labels = triple_barrier_labels(prices, pt_pct=0.05, sl_pct=0.05, max_holding=2)
    assert labels.iloc[0] == 0


def test_label_length_matches_input():
    prices = pd.Series(100 + np.random.default_rng(0).standard_normal(50).cumsum())
    labels = triple_barrier_labels(prices)
    assert len(labels) == len(prices)


def test_no_lookahead_on_last_row():
    # Son barda max_holding kadar veri olmaz → label 0 olmalı (bariyer vurulamaz)
    prices = pd.Series([100, 101, 102])
    labels = triple_barrier_labels(prices, pt_pct=0.10, sl_pct=0.10, max_holding=10)
    assert labels.iloc[-1] == 0
