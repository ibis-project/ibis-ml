from __future__ import annotations

import functools
import operator
from collections import Counter
from itertools import combinations_with_replacement
from typing import Any, Iterable

import ibis.expr.types as ir

from ibisml.core import Metadata, Step
from ibisml.select import SelectionType, selector


class PolynomialFeatures(Step):
    """A step for generating polynomial features.

    Parameters
    ----------
    inputs
        A selection of columns to generate polynomial features.
        All columns must be numeric.
    degree : int, default `2`
        The maximum degree of polynomial features to generate.

    Examples
    --------
    >>> import ibisml as ml

    Generate polynomial features for all numeric columns with a degree is 2.

    >>> step = ml.PolynomialFeatures(ml.numeric(), 2)

    Generate polynomial features a specific set of columns.

    >>> step = ml.PolynomialFeatures(["x", "y"], 2)
    """

    def __init__(self, inputs: SelectionType, *, degree: int = 2):
        if degree < 2:
            raise ValueError("Degree must be greater than 1")

        self.inputs = selector(inputs)
        self.degree = degree

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("degree", self.degree)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)

        for col_name in columns:
            if not isinstance(table[col_name], ir.NumericColumn):
                raise ValueError(
                    f"Cannot calculate polynomial features of {col_name!r} - "
                    "this column is not numeric"
                )
        combinations = []
        for d in range(2, self.degree + 1):
            combinations.extend(
                [
                    dict(Counter(comb))
                    for comb in combinations_with_replacement(columns, d)
                ]
            )
        self.combinations_ = combinations

    def transform_table(self, table: ir.Table) -> ir.Table:
        expressions = [
            functools.reduce(
                operator.mul,
                [
                    operator.pow(table[col], p) if p > 1 else table[col]
                    for col, p in combination.items()
                ],
            )
            for combination in self.combinations_
        ]
        return table.mutate(*expressions)
