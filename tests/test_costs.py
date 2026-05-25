"""BIST cost modeli — toplam bp doğru hesaplanıyor mu?"""
from backtest.costs import CostModel


def test_one_way_cost_components():
    m = CostModel(commission_pct=0.0005, bsmv_pct=0.05, spread_bps=10, slippage_bps=5)
    # 5bp komisyon + 0.25bp BSMV + 10bp spread + 5bp slippage = 20.25bp
    expected = 0.0005 + 0.0005 * 0.05 + 10/10000 + 5/10000
    assert abs(m.one_way_cost() - expected) < 1e-9


def test_round_trip_is_double():
    m = CostModel(commission_pct=0.001, bsmv_pct=0.05, spread_bps=20, slippage_bps=10)
    assert abs(m.round_trip_cost() - 2 * m.one_way_cost()) < 1e-9


def test_repr_includes_bp():
    m = CostModel(commission_pct=0.0005, bsmv_pct=0.05, spread_bps=10, slippage_bps=5)
    s = str(m)
    assert "bp" in s
    assert "one_way" in s
