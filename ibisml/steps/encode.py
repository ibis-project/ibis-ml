from __future__ import annotations

from collections import defaultdict
from typing import Any

import ibis
import ibis.expr.types as ir

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


def _compute_categories(
    table: ir.Table,
    columns: list[str],
    min_frequency: int | float | None = None,
    max_categories: int | None = None,
) -> dict[str, list[Any]]:
    import pandas as pd

    # We execute once for each type kind in the inputs. In the common case
    # (only string inputs) this means a single execution even for multiple
    # columns.
    groups = defaultdict(list)
    for c in columns:
        groups[table[c].type()].append(c)

    categories = {}

    if max_categories is not None or min_frequency is not None:

        def collect(col: str) -> ir.Table:
            query = (
                table.select(value=col)
                .group_by("value")
                .count("count")
                .mutate(column=ibis.literal(col))
                .order_by("count")
            )
            return query if max_categories is None else query.limit(max_categories)

        def process(df: pd.DataFrame) -> list[Any]:
            if isinstance(min_frequency, int):
                df = df[df["count"] >= min_frequency]
            elif isinstance(min_frequency, float):
                total = df["count"].sum()
                df = df[(df["count"] / total) >= min_frequency]

            return df["value"].sort_values().to_list()

    else:

        def collect(col: str) -> ir.Table:
            return (
                table.select(value=col, column=ibis.literal(col))
                .distinct()
                .order_by("value")
            )

        def process(df: pd.DataFrame) -> list[Any]:
            return df["value"].to_list()

    for group_type, group_cols in groups.items():
        query = ibis.union(*(collect(col) for col in group_cols))
        result_groups = query.execute().groupby("column")

        for col in group_cols:
            categories[col] = process(result_groups.get_group(col))

    return categories


class OneHotEncode(Step):
    def __init__(
        self,
        inputs: SelectionType,
        *,
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
    ):
        self.inputs = selector(inputs)
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def __repr__(self) -> str:
        return self._repr("inputs", min_frequency=None, max_categories=None)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)

        categories = _compute_categories(
            table, columns, self.min_frequency, self.max_categories
        )
        return ml.transforms.OneHotEncode(categories)


class CategoricalEncode(Step):
    def __init__(
        self,
        inputs: SelectionType,
        *,
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
    ):
        self.inputs = selector(inputs)
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def __repr__(self) -> str:
        return self._repr("inputs", min_frequency=None, max_categories=None)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)

        categories = _compute_categories(
            table, columns, self.min_frequency, self.max_categories
        )
        for col, cats in categories.items():
            metadata.set_categories(col, cats)
        return ml.transforms.CategoricalEncode(categories)
