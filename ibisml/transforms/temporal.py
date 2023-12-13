from __future__ import annotations

from typing import Literal

from ibisml.core import Transform

import ibis.expr.types as ir


class ExpandDateTime(Transform):
    def __init__(
        self,
        date_columns: list[str],
        date_components: list[Literal["day", "week", "month", "year", "dow", "doy"]],
        time_columns: list[str],
        time_components: list[Literal["hour", "minute", "second", "millisecond"]],
    ):
        self.date_columns = date_columns
        self.date_components = date_components
        self.time_columns = time_columns
        self.time_components = time_components

    @property
    def input_columns(self) -> list[str]:
        return self.date_columns + self.time_columns

    def transform(self, table: ir.Table) -> ir.Table:
        new_cols = []

        for name in self.date_columns:
            col = table[name]
            for comp in self.date_components:
                if comp == "day":
                    feat = col.day()
                elif comp == "week":
                    feat = col.week_of_year()
                elif comp == "month":
                    feat = col.month() - 1
                elif comp == "year":
                    feat = col.year()
                elif comp == "dow":
                    feat = col.day_of_week.index()
                elif comp == "doy":
                    feat = col.day_of_year()
                new_cols.append(feat.name(f"{name}_{comp}"))

        for name in self.time_columns:
            col = table[name]
            for comp in self.time_components:
                if comp == "hour":
                    feat = col.hour()
                elif comp == "minute":
                    feat = col.minute()
                elif comp == "second":
                    feat = col.second()
                elif comp == "millisecond":
                    feat = col.millisecond()
                new_cols.append(feat.name(f"{name}_{comp}"))

        return table.mutate(new_cols)


class ExpandDate(Transform):
    def __init__(
        self,
        columns: list[str],
        components: list[Literal["day", "week", "month", "year", "dow", "doy"]],
    ):
        self.columns = columns
        self.components = components

    @property
    def input_columns(self) -> list[str]:
        return self.columns

    def transform(self, table: ir.Table) -> ir.Table:
        new_cols = []
        for name in self.columns:
            col = table[name]
            for comp in self.components:
                if comp == "day":
                    feat = col.day()
                elif comp == "week":
                    feat = col.week_of_year()
                elif comp == "month":
                    feat = col.month() - 1
                elif comp == "year":
                    feat = col.year()
                elif comp == "dow":
                    feat = col.day_of_week.index()
                elif comp == "doy":
                    feat = col.day_of_year()
                new_cols.append(feat.name(f"{name}_{comp}"))
        return table.mutate(new_cols)


class ExpandTime(Transform):
    def __init__(
        self,
        columns: list[str],
        components: list[Literal["hour", "minute", "second", "millisecond"]],
    ):
        self.columns = columns
        self.components = components

    @property
    def input_columns(self) -> list[str]:
        return self.columns

    def transform(self, table: ir.Table) -> ir.Table:
        new_cols = []
        for name in self.columns:
            col = table[name]
            for comp in self.components:
                if comp == "hour":
                    feat = col.hour()
                elif comp == "minute":
                    feat = col.minute()
                elif comp == "second":
                    feat = col.second()
                elif comp == "millisecond":
                    feat = col.millisecond()
                new_cols.append(feat.name(f"{name}_{comp}"))
        return table.mutate(new_cols)
