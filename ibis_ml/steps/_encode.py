from __future__ import annotations

import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import ibis
import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector
from ibis_ml.steps._impute import FillNA

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "encoding"


def _compute_categories(
    table: ir.Table,
    columns: list[str],
    min_frequency: int | float | None = None,
    max_categories: int | None = None,
) -> dict[str, list[Any]]:
    if TYPE_CHECKING:
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

    for group_cols in groups.values():
        query = ibis.union(*(collect(col) for col in group_cols))
        result_groups = query.execute().groupby("column")

        for col in group_cols:
            categories[col] = process(result_groups.get_group(col))

    return categories


class OneHotEncode(Step):
    """A step for one-hot encoding select columns.

    The original input column is dropped, and N-category new columns are
    created with names like ``{input_column}_{category}``. Unknown categories
    will be ignored during transformation; the resulting one-hot encoded
    columns for this feature will be all zeros.

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
    >>> import ibis_ml as ml

    One-hot encode all string columns.

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
                categories[column] = list(range(len(cats)))
                metadata.drop_categories(column)
            else:
                to_compute.append(column)

        categories.update(
            _compute_categories(
                table, to_compute, self.min_frequency, self.max_categories
            )
        )

        self.categories_ = categories

    def transform_table(self, table: ir.Table) -> ir.Table:
        if not self.categories_:
            return table

        return table.mutate(
            [
                ibis.ifelse((table[col] == cat), 1, 0).name(f"{col}_{cat}")
                for col, cats in self.categories_.items()
                for cat in cats
            ]
        ).drop(*self.categories_)


class OrdinalEncode(Step):
    """A step for encoding select columns as integer arrays.

    Parameters
    ----------
    inputs
        A selection of columns to ordinal encode.
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
    >>> import ibis_ml as ml

    Ordinal encode all string columns.

    >>> step = ml.OrdinalEncode(ml.string())

    Ordinal encode a specific column, only including categories with at
    least 20 samples.

    >>> step = ml.OrdinalEncode("x", min_frequency=20)

    Ordinal encode a specific column, including at most 10 categories.

    >>> step = ml.OrdinalEncode("x", max_categories=10)
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
        columns = [
            column for column in columns if metadata.get_categories(column) is None
        ]
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
        for col, lookup in self.category_tables_.items():
            joined = table.left_join(
                lookup, table[col] == lookup[0], lname="{name}_left", rname=""
            )
            table = joined.drop(lookup.columns[0], f"{col}_left")

        return table


class CountEncode(Step):
    """A step for count encoding select columns.

    Parameters
    ----------
    inputs
        A selection of columns to count encode.

    Examples
    --------
    >>> import ibis_ml as ml

    Count encode all string columns.

    >>> step = ml.CountEncode(ml.string())
    """

    def __init__(self, inputs: SelectionType) -> None:
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)
        self._fit_expr = [table[c].value_counts() for c in columns]
        self.value_counts_ = {
            c: ibis.memtable(expr.to_pyarrow())
            for c, expr in zip(columns, self._fit_expr)
        }

        for c in columns:
            metadata.drop_categories(c)

    def transform_table(self, table: ir.Table) -> ir.Table:
        for c, value_counts in self.value_counts_.items():
            joined = table.left_join(
                value_counts, table[c] == value_counts[0], lname="left_{name}", rname=""
            )
            table = joined.drop(value_counts.columns[0], f"left_{c}").rename(
                {c: f"{c}_count"}
            )

        fillna = FillNA(self.value_counts_, 0)
        fillna.fit_table(table, Metadata())
        return fillna.transform_table(table)


class TargetEncode(Step):
    """A step for target encoding select columns.

    Parameters
    ----------
    inputs
        A selection of columns to target encode.
    smooth
        The amount of mixing of the target mean conditioned on the value of the
        category with the global target mean. A larger `smooth` value will put
        more weight on the global target mean.

    Examples
    --------
    >>> import ibis_ml as ml

    Target encode all string columns.

    >>> step = ml.TargetEncode(ml.string())
    """

    def __init__(self, inputs: SelectionType, smooth: float = 0.0) -> None:
        self.inputs = selector(inputs)
        self.smooth = smooth

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("smooth", self.smooth)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        self._fit_expr = []
        target_means_expr = table.aggregate(
            [table[c].mean().name(c) for c in metadata.targets]
        )
        self._fit_expr.append(target_means_expr)
        self.target_means_ = target_means_expr.execute().to_dict("records")[0]

        target_aggs = {}
        for target in metadata.targets:
            target_aggs[f"{target}_mean"] = table[target].mean()
            target_aggs[f"{target}_count"] = table[target].count()

        columns = self.inputs.select_columns(table, metadata)
        self.encodings_ = {}
        suffix = uuid.uuid4().hex[:6]
        for column in columns:
            agged = table.group_by(column).aggregate(**target_aggs)

            target_encodings = {}
            for target in metadata.targets:
                target_encodings[f"{target}_{suffix}"] = (
                    agged[f"{target}_mean"] * agged[f"{target}_count"]
                    + self.target_means_[target] * self.smooth
                ) / (agged[f"{target}_count"] + self.smooth)

            encoding_expr = agged.mutate(**target_encodings).drop(target_aggs)
            self._fit_expr.append(encoding_expr)
            self.encodings_[column] = ibis.memtable(encoding_expr.to_pyarrow())
            metadata.drop_categories(column)

    def transform_table(self, table: ir.Table) -> ir.Table:
        for c, encodings in self.encodings_.items():
            joined = table.left_join(
                encodings,
                (table[c] == encodings[0]) | table[c].isnull() & encodings[0].isnull(),  # noqa: PD003
                lname="left_{name}",
                rname="",
            )
            table = joined.drop(encodings.columns[0], f"left_{c}").rename(
                {c: encodings.columns[1]}
                if len(encodings.columns) < 3
                else {f"{c}{k + 1}": t for k, t in enumerate(encodings.columns[1:])}
            )

        for k, mean in enumerate(self.target_means_.values()):
            fillna = FillNA(
                (
                    c if len(self.target_means_) < 2 else f"{c}{k + 1}"
                    for c in self.encodings_
                ),
                mean,
            )
            fillna.fit_table(table, Metadata())
            table = fillna.transform_table(table)

        return table
