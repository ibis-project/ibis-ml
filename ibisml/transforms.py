from .core import Step

__all__ = (
    "StandardScaler",
    "OneHotEncoder",
)


class StandardScaler(Step):
    def __init__(self, *, center=True, scale=True, on_cols=None, on_type="numeric"):
        self.center = center
        self.scale = scale
        super().__init__(on_cols=on_cols, on_type=on_type)

    def fit(self, X, y=None):
        columns = self._select_columns(X)
        stats = []
        if self.center:
            stats.extend(X[c].mean().name(f"{c}_mean") for c in columns)
        if self.scale:
            stats.extend(X[c].std(how="pop").name(f"{c}_std") for c in columns)
        if stats:
            results = X.aggregate(stats).execute().to_dict("records")[0]

            if self.scale:
                scale = tuple(results[f"{c}_std"] for c in columns)
            else:
                scale = None

            if self.center:
                center = tuple(results[f"{c}_mean"] for c in columns)
            else:
                center = None

        self.input_columns_ = columns
        self.scales_ = scale
        self.centers_ = center

        return self

    def transform(self, X):
        if not self.center and not self.scale:
            return X

        out = [X[c] for c in self.input_columns_]
        if self.center:
            out = [x - center for (x, center) in zip(out, self.centers_)]
        if self.scale:
            out = [x / scale for (x, scale) in zip(out, self.scales_)]
        return X.mutate([x.name(c) for x, c in zip(out, self.input_columns_)])


class OneHotEncoder(Step):
    def __init__(self, *, on_cols=None, on_type="string"):
        super().__init__(on_cols=on_cols, on_type=on_type)

    def fit(self, X, y=None):
        columns = self._select_columns(X)
        categories = []
        for c in columns:
            categories.append(tuple(X.select(c).distinct().order_by(c).execute()[c]))

        self.input_columns_ = columns
        self.categories_ = tuple(categories)
        return self

    def transform(self, X):
        return X.mutate(
            [
                (X[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in zip(self.input_columns_, self.categories_)
                for cat in cats
            ]
        ).drop(*self.input_columns_)
