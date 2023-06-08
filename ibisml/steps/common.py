from __future__ import annotations

import ibis.expr.types as ir
import ibis.expr.datatypes as dt

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


class Drop(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def __repr__(self) -> str:
        return self._repr("inputs")

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Drop(columns)


class Cast(Step):
    def __init__(
        self,
        inputs: SelectionType,
        dtype: dt.DataType | type[dt.DataType] | str,
    ):
        self.inputs = selector(inputs)
        self.dtype = dt.dtype(dtype)

    def __repr__(self) -> str:
        return self._repr("inputs", "dtype")

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Cast(columns, self.dtype)
