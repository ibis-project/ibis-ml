from __future__ import annotations

import uuid
from typing import Any
from functools import cached_property

import ibis
import ibis.expr.types as ir

from ibisml.core import Transform


class OneHotEncode(Transform):
    """
    A transformation class that applies one-hot encoding to specified categorical columns.

    Examples
    ----------
    >>> from ibis.interactive import *
    >>> import ibisml as ml
    >>> penguins = ex.penguins.fetch()
    >>> recipe = ml.Recipe(ml.OneHotEncode("species"))
    >>> recipe.fit(penguins).transform(penguins).table.select(s.startswith("species"))
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
    ┃ species_Adelie ┃ species_Chinstrap ┃ species_Gentoo ┃
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
    │ int8           │ int8              │ int8           │
    ├────────────────┼───────────────────┼────────────────┤
    │              1 │                 0 │              0 │
    │              1 │                 0 │              0 │
    │              1 │                 0 │              0 │
    │              1 │                 0 │              0 │
    │              … │                 … │              … │
    └────────────────┴───────────────────┴────────────────┘
    """

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
            try:
                joined = table.left_join(
                    lookup,
                    table[col] == lookup[f"key_{self._rand_id}"],
                    lname="{name}_left",
                    rname="",
                )
            except TypeError:
                # Compat with ibis < 6
                joined = table.left_join(
                    lookup,
                    table[col] == lookup[f"key_{self._rand_id}"],
                    suffixes=("_left", ""),
                )
            table = joined.drop(f"key_{self._rand_id}", f"{col}_left")
        return table
