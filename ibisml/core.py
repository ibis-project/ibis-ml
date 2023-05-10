from __future__ import annotations

import operator
from functools import partial
from collections.abc import Iterable

import ibis
import ibis.expr.types as ir


_type_filters = {
    "numeric": lambda t: t.is_numeric(),
    "string": lambda t: t.is_string(),
    "int": lambda t: t.is_integer(),
    "float": lambda t: t.is_floating(),
    "timestamp": lambda t: t.is_timestamp(),
    "all": lambda t: True,
}


def normalize_X_y(X, y=None):
    if not isinstance(X, ir.Table):
        X = ibis.memtable(X)

    if y is None:
        table = X
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
    else:
        if isinstance(y, ir.Column):
            y_cols = (y.get_name(),)
        else:
            y_cols = tuple(y.columns)
        overlap = sorted(set(X.columns).intersection(y_cols))
        if overlap:
            raise ValueError(
                "Columns {overlap!r} in `y` have conflicting names with "
                "columns in X, please explicitly rename to no longer conflict"
            )
        if isinstance(y, ir.Column):
            table = X.mutate(y)
        else:
            table = X.mutate(y[c] for c in y_cols)

    return table, y_cols


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

    def select_columns(self, table, y_columns=None):
        x_columns = set(table.columns).difference(y_columns or ())
        out = set(x_columns)
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
                subset |= {c for c in x_columns if pred(schema[c])}

            out &= subset

        if self.on_cols is not None:
            if callable(self.on_cols):
                subset = {c for c in x_columns if self.on_cols(c)}
            else:
                on_cols = (
                    [self.on_cols] if isinstance(self.on_cols, str) else self.on_cols
                )
                missing = sorted(set(on_cols).difference(table.columns))
                if missing:
                    raise ValueError(f"Columns {missing!r} are missing from table")
                subset = set(on_cols)

            out &= subset

        return tuple(c for c in table.columns if c in out)

    def table_fit(self, table: ir.Table, y_columns: tuple[str] | None = None) -> Step:
        columns = self.select_columns(table, y_columns)
        self.do_fit(table, columns, y_columns)
        self.input_columns_ = columns
        return self

    def table_transform(self, table: ir.Table) -> ir.Table:
        return self.do_transform(table)

    def do_fit(
        self, table: ir.Table, columns: tuple[str], y_columns: tuple[str] | None = None
    ) -> None:
        raise NotImplementedError

    def do_transform(self, table: ir.Table) -> ir.Table:
        raise NotImplementedError

    def fit(self, X, y=None):
        table, y_columns = normalize_X_y(X, y)
        return self.table_fit(table, y_columns=y_columns)

    def transform(self, X):
        return self.table_transform(X)

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

    def _split_steps(self):
        steps = []
        ests = []
        for step in self.steps:
            if isinstance(step, Step):
                if ests:
                    raise ValueError(
                        "Cannot have ibisml steps in a pipeline after non-ibisml steps"
                    )
                steps.append(step)
            else:
                ests.append(step)
        return steps, ests

    def _fit_estimators(self, estimators, table, y_columns=None):
        exclude = set(y_columns)
        x_columns = [c for c in table.columns if c not in exclude]
        y_columns = list(y_columns)
        if all(hasattr(e, "partial_fit") for e in estimators):
            for batch in table.to_pyarrow_batches():
                df = batch.to_pandas(self_destruct=True)
                X = df[x_columns].to_numpy()
                y = df[y_columns].to_numpy()
                # TODO
        else:
            df = table.execute()
            X = df[x_columns].to_numpy()
            y = df[y_columns].to_numpy()
            if len(y_columns) == 1:
                y = y.ravel()

            last = estimators[-1]
            for est in estimators[:-1]:
                if hasattr(est, "fit_transform"):
                    est.fit_transform(X, y)
                else:
                    est.fit(X, y)
                    X = est.transform(X)
            last.fit(X, y)

    def fit(self, X, y=None):
        table, y_columns = normalize_X_y(X, y)

        steps, estimators = self._split_steps()

        for step in steps:
            step.table_fit(table, y_columns)
            table = step.table_transform(table)

        if estimators:
            self._fit_estimators(estimators, table, y_columns)

        return self

    def transform(self, X):
        steps, estimators = self._split_steps()
        if estimators and not hasattr(estimators[-1], "transform"):
            raise ValueError("`transform` is not available on this pipeline")

        table, _ = normalize_X_y(X)

        for step in steps:
            table = step.transform(table)

        if estimators:
            X = table.execute().to_numpy()
            for est in estimators:
                X = est.transform(X)
            return X
        else:
            return table

    def predict(self, X):
        steps, estimators = self._split_steps()
        if not estimators or not hasattr(estimators[-1], "predict"):
            raise ValueError("`predict` is not available on this pipeline")

        table, _ = normalize_X_y(X)

        for step in steps:
            table = step.transform(table)

        X = table.execute().to_numpy()
        last = estimators[-1]
        for est in estimators[:-1]:
            X = est.transform(X)
        return last.predict(X)
