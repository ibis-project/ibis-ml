from __future__ import annotations

from typing import Any, Iterable

import ibis.expr.types as ir

from ibis_ml.core import Metadata, Step
from ibis_ml.select import SelectionType, selector


class HandleUnivariateOutliers(Step):
    """A step for detecting and treating univariate outliers in numeric columns.
    Parameters
    ----------
    inputs
        A selection of columns to analyze for outliers. All columns must be numeric.
    method : {'z-score', 'IQR'}, default `z-score`
        The method to use for detecting outliers.
            - 'z-score' detects outliers the standard deviation from the mean
            for normally distributed data.
            - 'IQR' etects outliers using the interquartile range for skewed data.
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
            IQR = Q3 - Q1
            Upper Bound = Q3 + deviation_factor * IQR
            Lower Bound = Q1 - deviation_factor * IQR
    treatment : {'capping', 'trimming'}, default 'capping'
        The treatment to apply to the outliers. 'capping' replaces outlier values
        with the upper or lower bound, while 'trimming' removes outlier rows from
        the dataset.
    Examples
    --------
    >>> import ibis_ml as ml

    Capping outliers in all numeric columns using z-score method.

    >>> step = ml.HandleUnivariateOutliers(ml.numeric())

    Trimming outliers in a specific set of columns using IQR method.

    >>> step = ml.UnivariateOutlier(
        ["x", "y"],
        method="IQR",
        deviation_factor=2.0,
        treatment="trimming",
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
                f"Unsupported methods {method!r} encountered."
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
                upper_bound = right_bound + self.deviation_factor * distance
                lower_bound = left_bound - self.deviation_factor * distance
                stats[name] = {"upper_bound": upper_bound, "lower_bound": lower_bound}
        self.stats_ = stats

    def transform_table(self, table: ir.Table) -> ir.Table:
        if self.treatment == "capping":
            return table.mutate(
                **{
                    col_name: table[col_name].clip(
                        lower=stat["lower_bound"], upper=stat["upper_bound"]
                    )
                    for col_name, stat in self.stats_.items()
                }
            )
        else:
            return table.filter(
                [
                    (
                        (table[col_name] >= stat["lower_bound"])
                        & (table[col_name] <= stat["upper_bound"])
                        | (table[col_name].isnull() | table[col_name].isnan())  # noqa: PD003
                    )
                    for col_name, stat in self.stats_.items()
                ]
            )
