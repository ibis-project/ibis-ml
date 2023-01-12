from __future__ import annotations

import operator
from functools import partial
from collections.abc import Iterable

import ibis.expr.types as ir


_type_filters = {
    "numeric": lambda t: t.is_numeric(),
    "string": lambda t: t.is_string(),
    "int": lambda t: t.is_integer(),
    "float": lambda t: t.is_floating(),
    "timestamp": lambda t: t.is_timestamp(),
    "all": lambda t: True,
}


class Step:
    def __init__(self, *, on_cols=None, on_type=None):
        self.on_cols = on_cols
        self.on_type = on_type

    def __repr__(self):
        name = type(self).__name__
        parts = []

        if self.on_type is not None:
            parts.append(f"on_type={self.on_type!r}")
        if self.on_cols is not None:
            parts.append(f"on_cols={self.on_cols!r}")
        return f"{name}<{', '.join(parts)}>"

    def _normalize_X_y(self, X, y=None):
        if y is None:
            table = X
            X_cols = tuple(X.columns)
            y_cols = None
        elif isinstance(y, Iterable):
            if isinstance(y, str):
                y_cols = (y,)
            else:
                y_cols = tuple(y)
            for col in y_cols:
                if col not in X.columns:
                    raise ValueError(f"Column {col!r} does not exist")
            table = X
            X_cols = [c for c in table.columns if c not in y_cols]
        else:
            if isinstance(y, ir.Column):
                y_cols = (y.get_name(),)
            else:
                y_cols = tuple(y.columns)
            X_cols = tuple(X.columns)
            overlap = sorted(set(X_cols).intersection(y_cols))
            if overlap:
                raise ValueError(
                    "Columns {overlap!r} in `y` have conflicting names with "
                    "columns in X, please explicitly rename to no longer conflict"
                )
            if isinstance(y, ir.Column):
                table = X.mutate(y)
            else:
                table = X.mutate(y[c] for c in y_cols)

        return table, X_cols, y_cols

    def _select_columns(self, table, X_cols):
        cols = set(X_cols)
        if self.on_type is not None:
            if callable(self.on_type):
                preds = [self.on_type]
            else:
                on_type = (
                    [self.on_type] if isinstance(self.on_type, str) else self.on_type
                )
                preds = [
                    _type_filters[ot]
                    if isinstance(ot, str)
                    else partial(operator.eq, ot)
                    for ot in on_type
                ]
            subset = set()
            schema = table.schema()
            for pred in preds:
                subset |= {c for c in X_cols if pred(schema[c])}

            cols &= subset

        if self.on_cols is not None:
            if callable(self.on_cols):
                subset = {c for c in X_cols if self.on_cols(c)}
            else:
                on_cols = (
                    [self.on_cols] if isinstance(self.on_cols, str) else self.on_cols
                )
                missing = sorted(set(on_cols).difference(X_cols))
                if missing:
                    raise ValueError(f"Columns {missing!r} are missing from table")
                subset = set(on_cols)

            cols &= subset

        return tuple(c for c in X_cols if c in cols)

    def do_fit(
        self, table: ir.Table, columns: tuple[str], y_columns: tuple[str] | None = None
    ) -> ir.Table:
        pass

    def do_transform(self, table: ir.Table) -> ir.Table:
        pass

    def fit(self, X, y=None):
        table, x_columns, y_columns = self._normalize_X_y(X, y)
        columns = self._select_columns(table, x_columns)
        self.do_fit(table, columns, y_columns=y_columns)
        self.input_columns_ = columns
        return self

    def transform(self, X):
        return self.do_transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def __repr__(self):
        parts = "\n".join(f"- {s!r}" for s in self.steps)
        return f"Pipeline:\n{parts}"

    def fit_transform(self, X, y=None):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def fit(self, X, y=None):
        last_step = self.steps[-1]
        for step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        last_step.fit(X, y)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def predict(self, X):
        last_step = self.steps[-1]
        for step in self.steps[:-1]:
            X = step.transform(X)
        return last_step.predict(X)
