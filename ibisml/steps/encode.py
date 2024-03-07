from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Iterable

import ibis
import ibis.expr.types as ir

from ibisml.core import Metadata, Step
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
            return table.select(value=col, column=ibis.literal(col)).distinct()

        def process(df: pd.DataFrame) -> list[Any]:
            return df["value"].sort_values().to_list()

    for group_type, group_cols in groups.items():
        query = ibis.union(*(collect(col) for col in group_cols))
        result_groups = query.execute().groupby("column")

        for col in group_cols:
            categories[col] = process(result_groups.get_group(col))

    return categories


class OneHotEncode(Step):
    """A step for one-hot encoding select columns.

    The original input column is dropped, and N-category new columns are
    created with names like ``{input_column}_{category}``.

    Parameters
    ----------
    inputs
        A selection of columns to one-hot encode.
    min_frequency
        A minimum frequency of elements in the training set required to treat a
        column as a distinct category. May be either:

        - an integer, representing a minimum number of samples required.
        - a float in ``[0, 1]``, representing a minimum fraction of samples required.

        Defaults to ``None`` for no minimum frequency.
    max_categories
        A maximum number of categories to include. If set, only the most
        frequent ``max_categories`` categories are kept.

    Examples
    --------
    >>> import ibisml as ml

    One-hot encode all string columns

    >>> step = ml.OneHotEncode(ml.string())

    One-hot encode a specific column, only including categories with at least
    20 samples.

    >>> step = ml.OneHotEncode("x", min_frequency=20)

    One-hot encode a specific column, including at most 10 categories.

    >>> step = ml.OneHotEncode("x", max_categories=10)
    """

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

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        if self.min_frequency is not None:
            yield ("min_frequency", self.min_frequency)
        if self.max_categories is not None:
            yield ("max_categories", self.max_categories)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)

        categories = {}

        to_compute = []
        for column in columns:
            if cats := metadata.get_categories(column):
                categories[column] = list(range(len(cats.values)))
            else:
                to_compute.append(column)

        categories.update(
            _compute_categories(table, to_compute, self.min_frequency, self.max_categories)
        )

        self.categories_ = categories

    def transform_table(self, table: ir.Table) -> ir.Table:
        if not self.categories_:
            return table
        return table.mutate(
            [
                (table[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in self.categories_.items()
                for cat in cats
            ]
        ).drop(*self.categories_)


class CategoricalEncode(Step):
    """A step for categorical encoding select columns.

    Parameters
    ----------
    inputs
        A selection of columns to categorical encode.
    min_frequency
        A minimum frequency of elements in the training set required to treat a
        column as a distinct category. May be either:

        - an integer, representing a minimum number of samples required.
        - a float in ``[0, 1]``, representing a minimum fraction of samples required.

        Defaults to ``None`` for no minimum frequency.
    max_categories
        A maximum number of categories to include. If set, only the most
        frequent ``max_categories`` categories are kept.

    Examples
    --------
    >>> import ibisml as ml

    Categorical encode all string columns

    >>> step = ml.CategoricalEncode(ml.string())

    Categorical encode a specific column, only including categories with at
    least 20 samples.

    >>> step = ml.CategoricalEncode("x", min_frequency=20)

    Categorical encode a specific column, including at most 10 categories.

    >>> step = ml.CategoricalEncode("x", max_categories=10)
    """

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

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        if self.min_frequency is not None:
            yield ("min_frequency", self.min_frequency)
        if self.max_categories is not None:
            yield ("max_categories", self.max_categories)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        import pyarrow as pa  # type: ignore

        columns = self.inputs.select_columns(table, metadata)
        # Filter out already categorized columns
        columns = [column for column in columns if metadata.get_categories(column) is None]
        categories = _compute_categories(
            table, columns, self.min_frequency, self.max_categories
        )
        for col, cats in categories.items():
            metadata.set_categories(col, cats)

        tables = {}
        suffix = uuid.uuid4().hex[:6]
        for col, cats in categories.items():
            table = pa.Table.from_pydict(
                {f"key_{suffix}": cats, col: list(range(len(cats)))}
            )
            tables[col] = ibis.memtable(table, name=f"{col}_cats_{suffix}")

        self.category_tables_ = tables

    def transform_table(self, table: ir.Table) -> ir.Table:
        if not self.category_tables_:
            return table

        for col, lookup in self.category_tables_.items():
            joined = table.left_join(
                lookup,
                table[col] == lookup[0],
                lname="{name}_left",
                rname="",
            )
            table = joined.drop(lookup.columns[0], f"{col}_left")
        return table
