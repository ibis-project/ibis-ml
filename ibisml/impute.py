from .core import Step

__all__ = (
    "MeanImputer",
    "ModeImputer",
)


class _BaseImputer(Step):
    def __init__(self, *, on_cols=None, on_type=None):
        self.on_cols = on_cols
        self.on_type = on_type

    def _stat(self, col):
        pass

    def fit(self, X, y=None):
        columns = self._select_columns(X)
        stats = (
            X.aggregate([self._stat(X[c]).name(c) for c in columns])
            .execute()
            .to_dict("records")[0]
        )
        fill_values = tuple(stats[c] for c in columns)

        self.input_columns_ = columns
        self.fill_values_ = fill_values

        return self

    def transform(self, X):
        return X.mutate(
            [
                X[c].coalesce(v).name(c)
                for c, v in zip(self.input_columns_, self.fill_values_)
            ]
        )


class MeanImputer(_BaseImputer):
    def __init__(self, *, on_cols=None, on_type="numeric"):
        super().__init__(on_cols=on_cols, on_type=on_type)

    def _stat(self, col):
        return col.mean()


class ModeImputer(_BaseImputer):
    def __init__(self, on_cols=None, on_type="string"):
        super().__init__(on_cols=on_cols, on_type=on_type)

    def _stat(self, col):
        return col.mode()
