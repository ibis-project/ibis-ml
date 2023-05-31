from __future__ import annotations

from typing import Any

from ibisml.core import Transform

import ibis.expr.types as ir


class OneHotEncode(Transform):
    def __init__(self, categories: dict[str, list[Any]]):
        self.categories = categories

    def transform(self, table: ir.Table) -> ir.Table:
        if not self.categories:
            return table
        return table.mutate(
            [
                (table[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in self.categories.items()
                for cat in cats
            ]
        ).drop(*self.categories)
