from __future__ import annotations

from typing import Callable

from ibisml.core import Transform

import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.expr.deferred import Deferred


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


class MutateAt(Transform):
    def __init__(
        self,
        columns: list[str],
        expr: Callable[[ir.Column], ir.Column] | Deferred | None = None,
        named_exprs: dict[str, Callable[[ir.Column], ir.Column] | Deferred]
        | None = None,
    ):
        self.columns = columns
        self.expr = expr
        self.named_exprs = named_exprs or {}

    @property
    def input_columns(self) -> list[str]:
        return self.columns

    def transform(self, table: ir.Table) -> ir.Table:
        mutations: list[ir.Value] = []
        if self.expr is not None:
            func = self.expr.resolve if isinstance(self.expr, Deferred) else self.expr
            mutations.extend(
                func(table[c]).name(c) for c in self.columns  # type: ignore
            )
        for suffix, expr in self.named_exprs.items():
            func = expr.resolve if isinstance(expr, Deferred) else expr
            mutations.extend(
                func(table[c]).name(f"{c}_{suffix}") for c in self.columns  # type: ignore
            )
        return table.mutate(mutations)


class Mutate(Transform):
    def __init__(
        self,
        *exprs: Callable[[ir.Table], ir.Column] | Deferred,
        **named_exprs: Callable[[ir.Table], ir.Column] | Deferred,
    ):
        self.exprs = exprs
        self.named_exprs = named_exprs

    @property
    def input_columns(self) -> list[str]:
        # TODO: not all transforms have known input columns
        # need to rethink this interface
        return ["..."]

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(*self.exprs, **self.named_exprs)  # type: ignore
