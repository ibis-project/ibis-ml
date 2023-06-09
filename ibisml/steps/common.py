from __future__ import annotations

from typing import Callable, Iterable, Any

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.expr.deferred import Deferred

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


class Drop(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

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

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("", str(self.dtype))

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Cast(columns, self.dtype)


class MutateAt(Step):
    def __init__(
        self,
        inputs: SelectionType,
        _expr: Callable[[ir.Column], ir.Column] | Deferred | None = None,
        **named_exprs: Callable[[ir.Column], ir.Column] | Deferred,
    ):
        self.inputs = selector(inputs)
        self.expr = _expr
        self.named_exprs = named_exprs

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        if self.expr is not None:
            yield ("", self.expr)
        for name, expr in self.named_exprs.items():
            yield name, expr

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.MutateAt(
            columns, expr=self.expr, named_exprs=self.named_exprs
        )


class Mutate(Step):
    def __init__(
        self,
        *exprs: Callable[[ir.Table], ir.Column] | Deferred,
        **named_exprs: Callable[[ir.Table], ir.Column] | Deferred,
    ):
        self.exprs = exprs
        self.named_exprs = named_exprs

    def _repr(self) -> Iterable[tuple[str, Any]]:
        for expr in self.exprs:
            yield "", expr
        for name, expr in self.named_exprs.items():
            yield name, expr

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        return ml.transforms.Mutate(*self.exprs, **self.named_exprs)
