from __future__ import annotations

from typing import Any

from .core import Step, Transform
from .select import SelectionType, selector

import ibis.expr.types as ir


__all__ = (
    "Normalize",
    "OneHotEncode",
)


class NormalizeTransform(Transform):
    def __init__(self, stats: dict[str, tuple[float, float]]):
        self.stats = stats

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            ((table[c] - center) / scale).name(c)
            for c, (center, scale) in self.stats.items()
        )


class Normalize(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def fit(self, table: ir.Table, outcomes: list[str]) -> NormalizeTransform:
        columns = (self.inputs - outcomes).select_columns(table)

        stats = {}
        if columns:
            aggs = [table[c].mean().name(f"{c}_mean") for c in columns]
            aggs.extend(table[c].std(how="pop").name(f"{c}_std") for c in columns)
            results = table.aggregate(aggs).execute().to_dict("records")[0]
            for c in columns:
                stats[c] = (results[f"{c}_mean"], results[f"{c}_std"])
        return NormalizeTransform(stats)


class OneHotEncodeTransform(Transform):
    def __init__(self, categories: dict[str, list[Any]]):
        self.categories = categories

    def transform(self, table):
        if not self.categories:
            return table
        return table.mutate(
            [
                (table[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in self.categories.items()
                for cat in cats
            ]
        ).drop(*self.categories)


class OneHotEncode(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def fit(self, table: ir.Table, outcomes: list[str]) -> OneHotEncodeTransform:
        columns = (self.inputs - outcomes).select_columns(table)

        categories = {}
        for c in columns:
            categories[c] = list(table.select(c).distinct().order_by(c).execute()[c])

        return OneHotEncodeTransform(categories)
