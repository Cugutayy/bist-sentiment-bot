"""Walk-forward split mantığı — purge ve embargo doğru çalışıyor mu?"""
import pandas as pd

from backtest.walk_forward import make_splits


def test_splits_have_purge_gap():
    dates = pd.date_range("2022-01-01", "2026-01-01", freq="B")
    splits = make_splits(dates, train_window_days=180, test_window_days=30, purge_days=5, embargo_days=5)
    assert len(splits) > 0
    for s in splits:
        gap_days = (s.test_start - s.train_end).days
        assert gap_days >= 5, f"Purge gap eksik: {gap_days}"


def test_splits_have_embargo_between_iterations():
    dates = pd.date_range("2022-01-01", "2026-01-01", freq="B")
    splits = make_splits(dates, train_window_days=180, test_window_days=30, purge_days=5, embargo_days=10)
    for prev, curr in zip(splits, splits[1:]):
        embargo_days = (curr.train_start - prev.test_end).days
        assert embargo_days >= 10, f"Embargo eksik: {embargo_days}"


def test_test_window_is_correct_length():
    dates = pd.date_range("2022-01-01", "2026-01-01", freq="B")
    splits = make_splits(dates, train_window_days=180, test_window_days=21, purge_days=5, embargo_days=5)
    for s in splits:
        assert (s.test_end - s.test_start).days == 21


def test_no_splits_when_data_too_short():
    short_dates = pd.date_range("2026-01-01", "2026-01-15", freq="B")
    splits = make_splits(short_dates, train_window_days=180, test_window_days=30)
    assert splits == []
