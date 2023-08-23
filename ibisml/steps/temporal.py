from __future__ import annotations

from typing import Any, Iterable, Sequence, Literal

import ibis.expr.types as ir

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


class ExpandDate(Step):
    """A step for expanding date columns into one or more features.

    New features will be named ``{input_column}_{component}``. For example, if
    expanding a ``"year"`` component from column ``"x"``, the feature column
    would be named ``"x_year"``.

    Parameters
    ----------
    inputs
        A selection of date columns to expand into new features.
    components
        A sequence of components to expand. Options include

        - ``day``: the day of the month as a numeric value
        - ``week``: the week of the year as a numeric value
        - ``month``: the month of the year as a categorical value
        - ``year``: the year as a numeric value
        - ``dow``: the day of the week as a categorical value
        - ``doy``: the day of the year as a numeric value

        Defaults to ``["dow", "month", "year"]``.

    Examples
    --------
    >>> import ibisml as ml

    Expand date columns using the default components

    >>> step = ml.ExpandDate(ml.date())

    Expand specific columns using specific components

    >>> step = ml.ExpandDate(["x", "y"], ["day", "year"])
    """

    def __init__(
        self,
        inputs: SelectionType,
        components: Sequence[Literal["day", "week", "month", "year", "dow", "doy"]] = (
            "dow",
            "month",
            "year",
        ),
    ):
        self.inputs = selector(inputs)
        self.components = list(components)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("components", self.components)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        if "month" in self.components:
            for col in columns:
                metadata.set_categories(
                    f"{col}_month",
                    [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ],
                )
        if "dow" in self.components:
            for col in columns:
                metadata.set_categories(
                    f"{col}_dow",
                    [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thurday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ],
                )
        return ml.transforms.ExpandDate(columns, self.components)


class ExpandTime(Step):
    """A step for expanding time columns into one or more features.

    New features will be named ``{input_column}_{component}``. For example, if
    expanding an ``"hour"`` component from column ``"x"``, the feature column
    would be named ``"x_hour"``.

    Parameters
    ----------
    inputs
        A selection of time columns to expand into new features.
    components
        A sequence of components to expand. Options include ``hour``,
        ``minute``, ``second``, and ``millisecond``.

        Defaults to ``["hour", "minute", "second"]``.

    Examples
    --------
    >>> import ibisml as ml

    Expand time columns using the default components

    >>> step = ml.ExpandTime(ml.time())

    Expand specific columns using specific components

    >>> step = ml.ExpandTime(["x", "y"], ["hour", "minute"])
    """

    def __init__(
        self,
        inputs: SelectionType,
        components: Sequence[Literal["hour", "minute", "second", "millisecond"]] = (
            "hour",
            "minute",
            "second",
        ),
    ):
        self.inputs = selector(inputs)
        self.components = list(components)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("components", self.components)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.ExpandTime(columns, self.components)
