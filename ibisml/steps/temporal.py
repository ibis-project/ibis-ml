from __future__ import annotations

from typing import Any, Iterable, Sequence, Literal

import ibis.expr.types as ir

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


class ExpandDate(Step):
    def __init__(
        self,
        inputs: SelectionType,
        components: Sequence[Literal["day", "week", "month", "year", "dow", "doy"]] = (
            "dow",
            "month",
            "year",
        ),
    ):
        self.inputs = selector(inputs)
        self.components = list(components)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("components", self.components)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.ExpandDate(columns, self.components)


class ExpandTime(Step):
    def __init__(
        self,
        inputs: SelectionType,
        components: Sequence[Literal["hour", "minute", "second", "millisecond"]] = (
            "hour",
            "minute",
            "second",
        ),
    ):
        self.inputs = selector(inputs)
        self.components = list(components)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("components", self.components)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.ExpandTime(columns, self.components)
