from __future__ import annotations

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector

import ibis.expr.types as ir


class Drop(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def __repr__(self) -> str:
        return self._repr("inputs")

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Drop(columns)
