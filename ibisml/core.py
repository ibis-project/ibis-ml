from __future__ import annotations

from collections.abc import Iterable

import ibis
import ibis.expr.types as ir

from . import select


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
    def __init__(self, inputs):
        if isinstance(inputs, str):
            inputs = select.cols(inputs)
        elif isinstance(inputs, (list, tuple)):
            inputs = select.cols(*inputs)
        elif callable(inputs):
            inputs = select.where(inputs)
        elif not isinstance(inputs, select.Selector):
            raise TypeError(
                "inputs must be a str, list of strings, callable, or selector"
            )
        self.inputs = inputs

    def __repr__(self):
        name = type(self).__name__
        return f"{name}<{self.inputs}>"

    def determine_inputs(
        self, table: ir.Table, y_columns: tuple[str] | None = None
    ) -> tuple[str]:
        exclude = y_columns or ()
        return tuple(
            name
            for name in table.columns
            if self.inputs.matches(table[name])
            and name not in exclude
        )

    def table_fit(self, table: ir.Table, y_columns: tuple[str] | None = None) -> Step:
        input_columns = self.determine_inputs(table, y_columns)
        self.do_fit(table, input_columns, y_columns)
        self.input_columns_ = input_columns
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
