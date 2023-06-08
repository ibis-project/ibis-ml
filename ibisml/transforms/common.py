from __future__ import annotations

from ibisml.core import Transform

import ibis.expr.types as ir
import ibis.expr.datatypes as dt


class Drop(Transform):
    def __init__(self, columns: list[str]):
        self.columns = columns

    @property
    def input_columns(self) -> list[str]:
        return self.columns

    def transform(self, table: ir.Table) -> ir.Table:
        return table.drop(*self.columns)


class Cast(Transform):
    def __init__(self, columns: list[str], dtype: dt.DataType):
        self.columns = columns
        self.dtype = dtype

    @property
    def input_columns(self) -> list[str]:
        return self.columns

    def transform(self, table: ir.Table) -> ir.Table:
        return table.cast(dict.fromkeys(self.columns, self.dtype))
