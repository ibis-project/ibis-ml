from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "imputation"


def _fillna(col, val):
    if col.type().is_floating():
        return (col.isnull() | col.isnan()).ifelse(val, col)  # noqa: PD003
    else:
        return col.coalesce(val)


class FillNA(Step):
    """A step for filling NULL values in the input with a specific value.

    Parameters
    ----------
    inputs
        A selection of columns to fillna.
    fill_value
        The fill value to use. Must be castable to the dtype of all columns in
        inputs.

    Examples
    --------
    >>> import ibis_ml as ml

    Fill all NULL values in numeric columns with 0.

    >>> step = ml.FillNA(ml.numeric(), 0)

    Fill all NULL values in specific columns with 1.

    >>> step = ml.FillNA(["x", "y"], 1)
    """

    def __init__(self, inputs: SelectionType, fill_value: Any):
        self.inputs = selector(inputs)
        self.fill_value = fill_value

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("", self.fill_value)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        self.columns_ = self.inputs.select_columns(table, metadata)

    def transform_table(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [_fillna(table[c], self.fill_value).name(c) for c in self.columns_]
        )


class _BaseImpute(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def _stat(self, col: ir.Column) -> ir.Scalar:
        raise NotImplementedError

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)
        self._fit_expr = [
            table.aggregate([self._stat(table[c]).name(c) for c in columns])
        ]
        self.fill_values_ = self._fit_expr[0].execute().to_dict("records")[0]

    def transform_table(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [_fillna(table[c], v).name(c) for c, v in self.fill_values_.items()]
        )


class ImputeMean(_BaseImpute):
    """A step for replacing NULL values in select columns with their
    respective mean in the training set.

    Parameters
    ----------
    inputs
        A selection of columns to impute. All columns must be numeric.

    Examples
    --------
    >>> import ibis_ml as ml

    Replace NULL values in all numeric columns with their respective means,
    computed from the training dataset.

    >>> step = ml.ImputeMean(ml.numeric())
    """

    def _stat(self, col: ir.Column) -> ir.Scalar:
        if not isinstance(col, ir.NumericColumn):
            raise ValueError(
                f"Cannot compute mean of {col.get_name()} - "
                "this column is not numeric"
            )
        return col.mean()


class ImputeMedian(_BaseImpute):
    """A step for replacing NULL values in select columns with their
    respective medians in the training set.

    Parameters
    ----------
    inputs
        A selection of columns to impute. All columns must be numeric.

    Examples
    --------
    >>> import ibis_ml as ml

    Replace NULL values in all numeric columns with their respective medians,
    computed from the training dataset.

    >>> step = ml.ImputeMedian(ml.numeric())
    """

    def _stat(self, col: ir.Column) -> ir.Scalar:
        if not isinstance(col, ir.NumericColumn):
            raise ValueError(
                f"Cannot compute median of {col.get_name()} - "
                "this column is not numeric"
            )
        return col.median()


class ImputeMode(_BaseImpute):
    """A step for replacing NULL values in select columns with their
    respective modes in the training set.

    Parameters
    ----------
    inputs
        A selection of columns to impute.

    Examples
    --------
    >>> import ibis_ml as ml

    Replace NULL values in all numeric columns with their respective modes,
    computed from the training dataset.

    >>> step = ml.ImputeMode(ml.numeric())
    """

    def _stat(self, col: ir.Column) -> ir.Scalar:
        return col.mode()
