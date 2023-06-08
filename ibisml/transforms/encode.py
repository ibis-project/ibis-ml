from __future__ import annotations

import uuid
from typing import Any
from functools import cached_property

import ibis
import ibis.expr.types as ir

from ibisml.core import Transform


class OneHotEncode(Transform):
    def __init__(self, categories: dict[str, list[Any]]):
        self.categories = categories

    @property
    def input_columns(self) -> list[str]:
        return list(self.categories)

    def transform(self, table: ir.Table) -> ir.Table:
        if not self.categories:
            return table
        return table.mutate(
            [
                (table[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in self.categories.items()
                for cat in cats
            ]
        ).drop(*self.categories)


class CategoricalEncode(Transform):
    def __init__(self, categories: dict[str, list[Any]]):
        self.categories = categories
        # TODO: standardize IDs across steps/transforms
        self._rand_id = uuid.uuid4().hex[:6]

    @property
    def input_columns(self) -> list[str]:
        return list(self.categories)

    @cached_property
    def lookup_memtables(self):
        import pyarrow as pa  # type: ignore

        out = {}
        for col, cats in self.categories.items():
            table = pa.Table.from_pydict(
                {f"key_{self._rand_id}": cats, col: list(range(len(cats)))}
            )
            memtable = ibis.memtable(table, name=f"{col}_cats_{self._rand_id}")
            out[col] = memtable

        return out

    def transform(self, table: ir.Table) -> ir.Table:
        if not self.categories:
            return table

        for col, lookup in self.lookup_memtables.items():
            table = table.left_join(
                lookup,
                table[col] == lookup[f"key_{self._rand_id}"],
                lname="{name}_left",
                rname="",
            ).drop(f"key_{self._rand_id}", f"{col}_left")
        return table
