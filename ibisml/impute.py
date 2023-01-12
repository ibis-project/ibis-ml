from .core import Step

__all__ = (
    "MeanImputer",
    "ModeImputer",
)


class _BaseImputer(Step):
    def _stat(self, col):
        pass

    def do_fit(self, table, columns, y_columns=None):
        stats = (
            table.aggregate([self._stat(table[c]).name(c) for c in columns])
            .execute()
            .to_dict("records")[0]
        )
        self.fill_values_ = tuple(stats[c] for c in columns)

    def do_transform(self, table):
        return table.mutate(
            [
                table[c].coalesce(v).name(c)
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
