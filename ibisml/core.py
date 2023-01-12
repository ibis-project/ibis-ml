import operator
from functools import partial


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

    def _select_columns(self, X, y=None):
        cols = set(X.columns)
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
            for pred in preds:
                subset |= {c for c, t in X.schema().items() if pred(t)}

            cols &= subset

        if self.on_cols is not None:
            if callable(self.on_cols):
                subset = {c for c in X.columns if self.on_cols(c)}
            else:
                on_cols = (
                    [self.on_cols] if isinstance(self.on_cols, str) else self.on_cols
                )
                missing = sorted(set(on_cols).difference(X.columns))
                if missing:
                    raise ValueError(f"Columns {missing!r} are missing from table")
                subset = set(on_cols)

            cols &= subset

        return tuple(c for c in X.columns if c in cols)

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass

    def predict(self, X):
        pass

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
        last_step.fit(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def predict(self, X):
        for step in self.steps:
            X = step.predict(X)
        return X
