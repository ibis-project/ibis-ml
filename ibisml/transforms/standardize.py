from __future__ import annotations

from ibisml.core import Transform

import ibis.expr.types as ir


class ScaleMinMax(Transform):
    def __init__(self, stats: dict[str, tuple[float, float]]):
        self.stats = stats

    @property
    def input_columns(self) -> list[str]:
        return list(self.stats)

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [
                ((table[c] - min) / (max - min)).name(c)  # type: ignore
                for c, (max, min) in self.stats.items()
            ]
        )


class ScaleStandard(Transform):
    def __init__(self, stats: dict[str, tuple[float, float]]):
        self.stats = stats

    @property
    def input_columns(self) -> list[str]:
        return list(self.stats)

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [
                ((table[c] - center) / scale).name(c)  # type: ignore
                for c, (center, scale) in self.stats.items()
            ]
        )
