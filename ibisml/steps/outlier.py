from __future__ import annotations

from typing import Any, Iterable

import ibis
import ibis.expr.types as ir

from ibisml.core import Metadata, Step
from ibisml.select import SelectionType, selector


class UnivariateOutlier(Step):
    """A step for detecting and treating univariate outliers in numeric columns.

    Parameters
    ----------
    inputs
        A selection of columns to normalize. All columns must be numeric.
    method : {'z-score', 'IQR'}, default `z-score`
        The method to use for detecting outliers. 'z-score' calculates outliers
        based on the standard deviation from the mean for normally distributed data,
        while 'IQR' uses the interquartile range for skewed data.
    deviation_factor : int or float, default `3`
        The magnitude of deviation from the center is used to calculate
        the upper and lower bound for outlier detection.
        For z-score:
            Upper Bound: Mean + deviation_factor * standard deviation.
            Lower Bound: Mean - deviation_factor * standard deviation.
                - 68% of the data lies within 1 standard deviation.
                - 95% of the data lies within 2 standard deviations.
                - 99.7% of the data lies within 3 standard deviations.
        For IQR:
            IQR = Q3 -Q1
            Upper Bound = Q3 + deviation_factor * IQR
            Lower Bound = Q1 - deviation_factor * IQR
    treatment : {'capping', 'trimming'}, default 'capping'
        The treatment to apply to the outliers. 'capping' replaces outlier values
        with the upper or lower bound, while 'trimming' removes outlier rows from
        the dataset.

    Examples
    --------
    >>> import ibisml as ml

    Capping all numeric columns.

    >>> step = ml.UnivariateOutlier(ml.numeric())

    Trimming outliers in a specific set of columns using IQR method.

    >>> step = ml.UnivariateOutlier(
        ["x", "y"],
        method="IQR",
        deviation_factor=2.0,
        treatment="trimming"
    )
    """

    def __init__(
        self,
        inputs: SelectionType,
        *,
        method: str = "z-score",
        deviation_factor: int | float = 3,
        treatment: str = "capping",
    ):
        if method not in ["z-score", "IQR"]:
            raise ValueError(
                f"Unsupported method {method!r} encountered."
                "Supported methodes are 'z-score' and 'IQR'."
            )
        if treatment not in ["capping", "trimming"]:
            raise ValueError(
                f"Unsupported treatment {treatment!r} encountered."
                "Supported treatment are 'capping' and 'trimming'."
            )
        self.inputs = selector(inputs)
        self.method = method
        self.deviation_factor = deviation_factor
        self.treatment = treatment

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("method", self.method)
        yield ("deviation_factor", self.deviation_factor)
        yield ("treatment", self.treatment)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        columns = self.inputs.select_columns(table, metadata)
        stats = {}
        if columns:
            aggs = []
            for name in columns:
                c = table[name]
                if not isinstance(c, ir.NumericColumn):
                    raise ValueError(
                        f"Cannot be detect outlier for {name!r} - "
                        "this column is not numeric"
                    )
                if self.method == "z-score":
                    aggs.append(c.std().name(f"{name}_std"))
                    aggs.append(c.mean().name(f"{name}_mean"))
                elif self.method == "IQR":
                    aggs.append(c.quantile(0.25).name(f"{name}_25"))
                    aggs.append(c.quantile(0.75).name(f"{name}_75"))

            results = table.aggregate(aggs).execute().to_dict("records")[0]

            for name in columns:
                if self.method == "z-score":
                    left_bound = right_bound = results[f"{name}_mean"]
                    distance = results[f"{name}_std"]  # std
                elif self.method == "IQR":
                    left_bound = results[f"{name}_25"]
                    right_bound = results[f"{name}_75"]
                    distance = right_bound - left_bound  # IQR
                upper = right_bound + self.deviation_factor * distance
                lower = left_bound - self.deviation_factor * distance
                stats[name] = {"upper_bound": upper, "lower_bound": lower}
        self.stats_ = stats

    def transform_table(self, table: ir.Table) -> ir.Table:
        expressions = {}
        predicates = []
        for col_name, stat in self.stats_.items():
            upper_bound = stat["upper_bound"]
            lower_bound = stat["lower_bound"]
            col = table[col_name]
            if self.treatment == "capping":
                capped_col = ibis.case()
                capped_col = capped_col.when(col > upper_bound, upper_bound)
                capped_col = capped_col.when(col < lower_bound, lower_bound)
                capped_col = capped_col.else_(col).end()
                expressions[col_name] = capped_col
            elif self.treatment == "trimming":
                predicates.append(
                    ((col >= lower_bound) & (col <= upper_bound)) | (col.isnull())
                )
        return table.mutate(**expressions).filter(predicates)
