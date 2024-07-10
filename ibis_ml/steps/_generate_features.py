from __future__ import annotations

import functools
import operator
from collections import Counter
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any

import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector

if TYPE_CHECKING:
    from collections.abc import Iterable

_DOCS_PAGE_NAME = "feature-generation"


class CreatePolynomialFeatures(Step):
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
    >>> import ibis_ml as ml

    Generate polynomial features for all numeric columns with a degree is 2.

    >>> step = ml.CreatePolynomialFeatures(ml.numeric(), degree=2)

    Generate polynomial features a specific set of columns.

    >>> step = ml.CreatePolynomialFeatures(["x", "y"], degree=2)
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
                [Counter(comb) for comb in combinations_with_replacement(columns, d)]
            )
        self.combinations_ = combinations

    def transform_table(self, table):
        expressions = {}
        for combination in self.combinations_:
            exp = functools.reduce(
                operator.mul,
                [
                    table[col] ** p if p > 1 else table[col]
                    for col, p in combination.items()
                ],
            )
            name = "poly_" + "_".join(
                f"{col}^{p}" if p > 1 else col for col, p in combination.items()
            )
            expressions[name] = exp

        return table.mutate(**expressions)
