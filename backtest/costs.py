"""BIST için gerçekçi işlem maliyeti modeli.

BIST gerçeği:
- Komisyon: %0.05 (~5bp) — broker'a göre değişir
- BSMV: komisyonun %5'i (yani 5bp komisyon → 0.25bp ek vergi)
- Spread: likit hisselerde 5-10bp, küçük caplerde 25bp+
- Slippage: market emir + büyük pozisyon → +5-15bp impact
- Toplam tek yön: ~15-30bp
- Round-trip: ~30-60bp
"""
from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS


@dataclass(frozen=True)
class CostModel:
    """Yapılandırılabilir cost modeli — settings.yaml costs bloğundan okur."""
    commission_pct: float
    bsmv_pct: float
    spread_bps: float
    slippage_bps: float

    @classmethod
    def from_settings(cls) -> "CostModel":
        c = SETTINGS["costs"]
        return cls(
            commission_pct=c["commission_pct"],
            bsmv_pct=c["bsmv_pct"],
            spread_bps=c["spread_bps"],
            slippage_bps=c["slippage_bps"],
        )

    def one_way_cost(self) -> float:
        """Tek yön toplam maliyet (decimal — yani 0.0015 = %0.15)."""
        comm = self.commission_pct
        bsmv = comm * self.bsmv_pct
        spread = self.spread_bps / 10_000
        slip = self.slippage_bps / 10_000
        return comm + bsmv + spread + slip

    def round_trip_cost(self) -> float:
        """Al-sat çevriminin toplam maliyeti."""
        return 2 * self.one_way_cost()

    def __str__(self) -> str:
        ow = self.one_way_cost() * 10_000
        return f"CostModel(one_way={ow:.1f}bp, round_trip={ow*2:.1f}bp)"


# Hızlı kullanım için default instance
DEFAULT = CostModel.from_settings()
