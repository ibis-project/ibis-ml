from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "standardization"


class ScaleMinMax(Step):
    """A step for normalizing selected numeric columns to have a maximum value of 1
    and a minimum value of 0.

    Parameters
    ----------
    inputs
        A selection of columns to normalize. All columns must be numeric.

    Examples
    --------
    >>> import ibis_ml as ml

    Normalize all numeric columns.

    >>> step = ml.ScaleMinMax(ml.numeric())

    Normalize a specific set of columns.

    >>> step = ml.ScaleMinMax(["x", "y"])
    """

    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)

        stats = {}
        if columns:
            aggs = []
            for name in columns:
                c = table[name]
                if not isinstance(c, ir.NumericColumn):
                    raise ValueError(
                        f"Cannot be normalized {name!r} - this column is not numeric"
                    )

                aggs.append(c.max().name(f"{name}_max"))
                aggs.append(c.min().name(f"{name}_min"))

            expr = table.aggregate(aggs)
            self._fit_expr = [expr]
            results = expr.execute().to_dict("records")[0]
            for name in columns:
                stats[name] = (results[f"{name}_max"], results[f"{name}_min"])

        self.stats_ = stats

    def transform_table(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [
                ((table[c] - min) / (max - min)).name(c)  # type: ignore
                for c, (max, min) in self.stats_.items()
            ]
        )


class ScaleStandard(Step):
    """A step for normalizing select numeric columns to have a standard
    deviation of one and a mean of zero.

    Parameters
    ----------
    inputs
        A selection of columns to normalize. All columns must be numeric.

    Examples
    --------
    >>> import ibis_ml as ml

    Normalize all numeric columns.

    >>> step = ml.ScaleStandard(ml.numeric())

    Normalize a specific set of columns.

    >>> step = ml.ScaleStandard(["x", "y"])
    """

    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)

        stats = {}
        if columns:
            aggs = []
            for name in columns:
                c = table[name]
                if not isinstance(c, ir.NumericColumn):
                    raise ValueError(
                        f"Cannot standardize {name!r} - this column is not numeric"
                    )

                aggs.append(c.mean().name(f"{name}_mean"))
                aggs.append(c.std(how="pop").name(f"{name}_std"))

            self._fit_expr = [table.aggregate(aggs)]
            results = self._fit_expr[-1].execute().to_dict("records")[0]
            for name in columns:
                stats[name] = (results[f"{name}_mean"], results[f"{name}_std"])

        self.stats_ = stats

    def transform_table(self, table: ir.Table) -> ir.Table:
        return table.mutate(
            [
                ((table[c] - center) / scale).name(c)  # type: ignore
                for c, (center, scale) in self.stats_.items()
            ]
        )
