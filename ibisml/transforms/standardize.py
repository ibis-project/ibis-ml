from __future__ import annotations

from ibisml.core import Transform

import ibis.expr.types as ir


class ScaleStandard(Transform):
    def __init__(self, stats: dict[str, tuple[float, float]]):
        self.stats = stats

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [
                ((table[c] - center) / scale).name(c)  # type: ignore
                for c, (center, scale) in self.stats.items()
            ]
        )
