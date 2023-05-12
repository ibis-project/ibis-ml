from .core import Step

__all__ = (
    "StandardScaler",
    "OneHotEncoder",
)


class StandardScaler(Step):
    def __init__(self, inputs, *, center=True, scale=True):
        self.center = center
        self.scale = scale
        super().__init__(inputs)

    def do_fit(self, table, columns, y_columns=None):
        stats = []
        if self.center:
            stats.extend(table[c].mean().name(f"{c}_mean") for c in columns)
        if self.scale:
            stats.extend(table[c].std(how="pop").name(f"{c}_std") for c in columns)
        if stats:
            results = table.aggregate(stats).execute().to_dict("records")[0]

            if self.scale:
                scale = tuple(results[f"{c}_std"] for c in columns)
            else:
                scale = None

            if self.center:
                center = tuple(results[f"{c}_mean"] for c in columns)
            else:
                center = None

        self.scales_ = scale
        self.centers_ = center

    def do_transform(self, table):
        if not self.center and not self.scale:
            return table

        out = [table[c] for c in self.input_columns_]
        if self.center:
            out = [x - center for (x, center) in zip(out, self.centers_)]
        if self.scale:
            out = [x / scale for (x, scale) in zip(out, self.scales_)]
        return table.mutate([x.name(c) for x, c in zip(out, self.input_columns_)])


class OneHotEncoder(Step):
    def do_fit(self, table, columns, y_columns=None):
        categories = []
        for c in columns:
            categories.append(
                tuple(table.select(c).distinct().order_by(c).execute()[c])
            )

        self.categories_ = tuple(categories)

    def do_transform(self, table):
        return table.mutate(
            [
                (table[col] == cat).cast("int8").name(f"{col}_{cat}")
                for col, cats in zip(self.input_columns_, self.categories_)
                for cat in cats
            ]
        ).drop(*self.input_columns_)
