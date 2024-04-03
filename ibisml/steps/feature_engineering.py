from __future__ import annotations

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

        non_numeric_cols = [
            col for col in columns if not isinstance(table[col], ir.NumericColumn)
        ]
        if non_numeric_cols:
            raise ValueError(
                "Cannot fit polynomial features step: "
                f"{[c for c in non_numeric_cols]} is not numeric"
            )
        combinations = []
        for d in range(2, self.degree + 1):
            combinations.extend(combinations_with_replacement(columns, d))
        self.combinations_ = combinations

    def transform_table(self, table: ir.Table) -> ir.Table:
        expressions = []
        for combination in self.combinations_:
            expression = 1
            for column in combination:
                expression *= table[column]
            expressions.append(
                expression.name(f"poly_{'_'.join(column for column in combination)}")
            )

        return table.mutate(**{exp.get_name(): exp for exp in expressions})
