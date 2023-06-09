from __future__ import annotations

from typing import Any, Iterable

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector

import ibis.expr.types as ir


class FillNA(Step):
    def __init__(self, inputs: SelectionType, fill_value: Any):
        self.inputs = selector(inputs)
        self.fill_value = fill_value

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("", self.fill_value)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.FillNA({c: self.fill_value for c in columns})


class _BaseImpute(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def _stat(self, col: ir.Column) -> ir.Scalar:
        raise NotImplementedError

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)

        stats = (
            table.aggregate([self._stat(table[c]).name(c) for c in columns])
            .execute()
            .to_dict("records")[0]
        )
        return ml.transforms.FillNA(stats)


class ImputeMean(_BaseImpute):
    def _stat(self, col: ir.Column) -> ir.Scalar:
        if not isinstance(col, ir.NumericColumn):
            raise ValueError(
                f"Cannot compute mean of {col.get_name()} - "
                "this column is not numeric"
            )
        return col.mean()


class ImputeMedian(_BaseImpute):
    def _stat(self, col: ir.Column) -> ir.Scalar:
        if not isinstance(col, ir.NumericColumn):
            raise ValueError(
                f"Cannot compute median of {col.get_name()} - "
                "this column is not numeric"
            )
        return col.median()


class ImputeMode(_BaseImpute):
    def _stat(self, col: ir.Column) -> ir.Scalar:
        return col.mode()
