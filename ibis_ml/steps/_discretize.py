from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis
import ibis.expr.types as ir
import numpy as np

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "discretization"


class DiscretizeKBins(Step):
    """A step for binning numeric data into intervals.

    Parameters
    ----------
    inputs
        A selection of columns to bin.
    n_bins : int, default=5
        Number of bins to create.
    strategy : str, {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the bin edges.
        - 'uniform': Evenly spaced bins between the minimum and maximum values.
        - 'quantile': Bins are created based on data quantiles.


    Raises
    ----------
    ValueError
        If `n_bins` is less than or equal to 1 or if an unsupported
        `strategy` is provided.

    Examples
    --------
    >>> import ibis
    >>> import ibis_ml as ml
    >>> from ibis_ml.core import Metadata
    >>> ibis.options.interactive = True

    Load penguins dataset

    >>> p = ibis.examples.penguins.fetch()

    Bin all numeric columns.

    >>> step = ml.DiscretizeKBins(ml.numeric(), n_bins=10)
    >>> step.fit_table(p, Metadata())
    >>> step.transform_table(p)

    Bin specific numeric columns.

    >>> step = ml.DiscretizeKBins(["bill_length_mm"], strategy="quantile")
    >>> step.fit_table(p, Metadata())
    >>> step.transform_table(p)
    """

    def __init__(
        self, inputs: SelectionType, *, n_bins: int = 5, strategy: str = "uniform"
    ):
        if n_bins <= 1:
            raise ValueError("Number of bins must be greater than 1.")

        if strategy not in ["uniform", "quantile"]:
            raise ValueError(
                f"Unsupported strategy {strategy!r} encountered."
                "Supported strategies are 'uniform' and 'quantile'."
            )

        self.inputs = selector(inputs)
        self.n_bins = n_bins
        self.strategy = strategy

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("n_bins", self.n_bins)
        yield ("strategy", self.strategy)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)
        bins_edge = {}
        if columns:
            if self.strategy == "uniform":
                bins_edge = self._fit_uniform_strategy(table, columns)
            elif self.strategy == "quantile":
                bins_edge = self._fit_quantile_strategy(table, columns)
        self.bins_edge_ = bins_edge

    def _fit_uniform_strategy(
        self, table: ir.Table, columns: list[str]
    ) -> dict[str, list[float]]:
        aggs = []
        for col_name in columns:
            col = table[col_name]
            if not isinstance(col, ir.NumericColumn):
                raise ValueError(
                    f"Cannot discretize {col_name!r} - this column is not numeric"
                )
            aggs.append(col.max().name(f"{col_name}_max"))
            aggs.append(col.min().name(f"{col_name}_min"))

        self._fit_expr = [table.aggregate(aggs)]
        results = self._fit_expr[-1].execute().to_dict("records")[0]

        return {
            col_name: np.linspace(
                results[f"{col_name}_min"], results[f"{col_name}_max"], self.n_bins + 1
            )
            for col_name in columns
        }

    def _fit_quantile_strategy(
        self, table: ir.Table, columns: list[str]
    ) -> dict[str, list[float]]:
        aggs = []
        percentiles = np.linspace(0, 1, self.n_bins + 1)
        for col_name in columns:
            col = table[col_name]
            if not isinstance(col, ir.NumericColumn):
                raise ValueError(
                    f"Cannot discretize {col_name!r} - this column is not numeric"
                )
            aggs.extend([col.quantile(q).name(f"{col_name}_{q}") for q in percentiles])

        self._fit_expr = [table.aggregate(aggs)]
        results = self._fit_expr[-1].execute().to_dict("records")[0]

        return {
            col_name: [results[f"{col_name}_{q}"] for q in percentiles]
            for col_name in columns
        }

    def transform_table(self, table: ir.Table) -> ir.Table:
        aggs = []
        for col_name, edges in self.bins_edge_.items():
            edges = edges[1:-1]
            col = table[col_name]
            case_builder = ibis.case()
            if len(edges) >= 1:
                case_builder = case_builder.when(col <= edges[0], 0)
                case_builder = case_builder.when(col > edges[-1], len(edges))
            for i, cutoff in enumerate(edges):
                if i == 0:
                    continue
                prev_cutoff = edges[i - 1]
                case_builder = case_builder.when(
                    (col > prev_cutoff) & (col <= cutoff), i
                )
            case_builder = case_builder.end()
            col_name = f"{col_name}_{self.n_bins}_bin_{self.strategy}"
            aggs.append({col_name: case_builder})

        return table.mutate(
            **{name: expr for agg in aggs for name, expr in agg.items()}
        )
