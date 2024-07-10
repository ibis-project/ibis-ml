from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "feature-selection"


class DropZeroVariance(Step):
    """A step for removing columns with zero variance.

    Parameters
    ----------
    inputs : SelectionType
        A selection of columns to analyze for zero variance.
    tolerance : int | float, optional
        Tolerance level for considering variance as zero.
        Columns with variance less than this tolerance will be removed.
        Default is 1e-4.

    Examples
    --------
    >>> import ibis_ml as ml

    To remove columns with zero variance:
    >>> step = ml.DropZeroVariance(ml.everything())

    To remove all numeric columns with zero variance:
    >>> step = ml.DropZeroVariance(ml.numeric())

    To remove all string or categorical columns with only one unique value:
    >>> step = ml.DropZeroVariance(ml.nominal())
    """

    def __init__(self, inputs: SelectionType, *, tolerance: int | float = 1e-4):
        self.inputs = selector(inputs)
        self.tolerance = tolerance

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("tolerance", self.tolerance)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)
        cols = []
        if columns:
            aggs = []
            for name in columns:
                c = table[name]
                if isinstance(c, ir.NumericColumn):
                    # Compute variance for numeric columns
                    aggs.append(c.var().name(f"{name}_var"))
                else:
                    # Compute unique count for non-numeric columns
                    # NULL value is not counted in nunique()
                    aggs.append(c.nunique().name(f"{name}_var"))

            self._fit_expr = [table.aggregate(aggs)]
            results = self._fit_expr[0].execute().to_dict("records")[0]
            for col_name in columns:
                col_var = results[f"{col_name}_var"]
                if isinstance(table[col_name], ir.NumericColumn):
                    # Check variance for numeric columns
                    if math.isnan(col_var) or col_var < self.tolerance:
                        cols.append(col_name)
                elif col_var < 2:
                    # Check unique count for non-numeric columns
                    cols.append(col_name)
                    metadata.drop_categories(col_name)

        self.cols_ = cols

    def transform_table(self, table: ir.Table) -> ir.Table:
        return table.drop(*self.cols_)
