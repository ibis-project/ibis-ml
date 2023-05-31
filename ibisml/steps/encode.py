from __future__ import annotations

import ibisml as ml
from ibisml.core import Step, Transform
from ibisml.select import SelectionType, selector

import ibis.expr.types as ir

__all__ = ("OneHotEncode",)


class OneHotEncode(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def fit(self, table: ir.Table, outcomes: list[str]) -> Transform:
        columns = (self.inputs - outcomes).select_columns(table)

        categories = {}
        for c in columns:
            categories[c] = list(table.select(c).distinct().order_by(c).execute()[c])

        return ml.transforms.OneHotEncode(categories)
