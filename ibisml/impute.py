from __future__ import annotations

from typing import Any

from .core import Step, Transform
from .select import SelectionType, selector

import ibis.expr.types as ir

__all__ = (
    "FillNA",
    "ImputeMean",
    "ImputeMode",
)


class FillNA(Transform):
    def __init__(self, fill_values: dict[str, Any]):
        self.fill_values = fill_values

    def transform(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [table[c].coalesce(v).name(c) for c, v in self.fill_values.items()]
        )


class _BaseImpute(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _stat(self, col: ir.Column) -> ir.Value:
        raise NotImplementedError

    def fit(self, table: ir.Table, outcomes: list[str]) -> FillNA:
        columns = (self.inputs - outcomes).select_columns(table)

        stats = (
            table.aggregate([self._stat(table[c]).name(c) for c in columns])
            .execute()
            .to_dict("records")[0]
        )
        return FillNA(stats)


class ImputeMean(_BaseImpute):
    def _stat(self, col: ir.Column) -> ir.Value:
        return col.mean()


class ImputeMode(_BaseImpute):
    def _stat(self, col: ir.Column) -> ir.Value:
        return col.mode()
