from __future__ import annotations

from typing import Any, Iterable

import ibis
import ibis.expr.types as ir
import numpy as np

from ibisml.core import Metadata, Step
from ibisml.select import SelectionType, selector


class KBinsDiscretizer(Step):
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
    overwrite : bool, default=False
        Whether to overwrite existing columns or create new ones.

    Raises
    ----------
    ValueError
        If `n_bins` is less than or equal to 1 or if an unsupported
        `strategy` is provided.

    Examples
    --------
    >>> import ibis
    >>> import ibisml as ml
    >>> from ibisml.core import Metadata
    >>> ibis.options.interactive = True

    Load penguins dataset

    >>> p = ibis.examples.penguins.fetch()

    Bin all numeric columns.

    >>> step = ml.KBinsDiscretizer(ml.numeric(), n_bins=10, overwrite=True)
    >>> step.fit_table(p, Metadata())
    >>> step.transform_table(p)

    Bin specific numeric columns.

    >>> step = ml.KBinsDiscretizer(
        ["bill_length_mm"],
        strategy="quantile",
        overwrite="True",
    )
    >>> step.fit_table(p, Metadata())
    >>> step.transform_table(p)
    """

    def __init__(
        self,
        inputs: SelectionType,
        *,
        n_bins: int = 5,
        strategy: str = "uniform",
        overwrite: bool = False,
    ):
        if n_bins <= 1:
            raise ValueError("Number of n_bins must be greater than 1.")

        if strategy not in ["uniform", "quantile"]:
            raise ValueError(
                f"Unsupported strategy '{self.strategy}' encountered."
                "Supported strategies are 'uniform' and 'quantile'."
            )

        self.inputs = selector(inputs)
        self.n_bins = n_bins
        self.strategy = strategy
        self.overwrite = overwrite

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("n_bins", self.n_bins)
        yield ("strategy", self.strategy)
        yield ("overwrite", self.overwrite)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        """
        Bin continuous data into intervals.
        """
        columns = self.inputs.select_columns(table, metadata)
        if columns:
            if self.strategy == "uniform":
                self._fit_uniform_strategy(table, columns)
            elif self.strategy == "quantile":
                self._fit_quantile_strategy(table, columns)
        else:
            raise ValueError(f"No columns are selected: {self.inputs}")

    def _fit_uniform_strategy(self, table: ir.Table, columns: list[str]) -> None:
        aggs = []
        for col_name in columns:
            c = table[col_name]
            if not isinstance(c, ir.NumericColumn):
                raise ValueError(f"Cannot discretize non-numeric column: '{col_name}'.")
            aggs.append(c.max().name(f"{col_name}_max"))
            aggs.append(c.min().name(f"{col_name}_min"))

        bins_edge = {}
        results = table.aggregate(aggs).execute().to_dict("records")[0]
        for col_name in columns:
            edges = np.linspace(
                results[f"{col_name}_min"], results[f"{col_name}_max"], self.n_bins + 1
            )
            bins_edge[col_name] = edges

        self.bins_edge_ = bins_edge

    def _fit_quantile_strategy(self, table: ir.Table, columns: list[str]) -> None:
        aggs = []
        percentiles = np.linspace(0, 1, self.n_bins + 1)
        for col_name in columns:
            if not isinstance(table[col_name], ir.NumericColumn):
                raise ValueError(f"Cannot discretize non-numeric column: '{col_name}'.")

            aggs.extend(
                [
                    table[col_name].quantile(q).name(f"{col_name}_{q}")
                    for q in percentiles
                ]
            )

        results = table.aggregate(aggs).execute().to_dict("records")[0]
        bins_edge = {
            col_name: [results[f"{col_name}_{q}"] for q in percentiles]
            for col_name in columns
        }

        self.bins_edge_ = bins_edge

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
            if not self.overwrite:
                col_name = f"{col_name}_{self.n_bins}_bin_{self.strategy}"
            aggs.append({col_name: case_builder})

        return table.mutate(
            **{name: expr for agg in aggs for name, expr in agg.items()}
        )
