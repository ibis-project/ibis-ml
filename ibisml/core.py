from __future__ import annotations

from collections.abc import Sequence, Iterable
from typing import Any, cast, TYPE_CHECKING

import pyarrow as pa
import ibis.expr.types as ir

if TYPE_CHECKING:
    import pandas as pd


class Categories:
    def __init__(self, values: pa.Array, ordered: bool = False):
        self.values = values
        self.ordered = ordered


class Metadata:
    def __init__(self, outcomes: list[str] | None = None):
        self.outcomes = outcomes or []
        self.categories: dict[str, Categories] = {}

    def get_categories(self, column: str) -> Categories | None:
        return self.categories.get(column)

    def set_categories(
        self, column: str, values: pa.Array | list[Any], ordered: bool = False
    ) -> None:
        self.categories[column] = Categories(pa.array(values), ordered)

    def drop_categories(self, column: str) -> None:
        self.categories.pop(column, None)


def _categorize_wrap_reader(
    reader: pa.RecordBatchReader, categories: dict[str, Categories]
) -> Iterable[pa.RecordBatch]:
    for batch in reader:
        out = {}
        for name, col in zip(batch.schema.names, batch.columns):
            cats = categories.get(name)
            if cats is not None:
                col = pa.DictionaryArray.from_arrays(
                    col, cats.values, ordered=cats.ordered
                )
            out[name] = col
        yield pa.RecordBatch.from_pydict(out)


class Dataset:
    def __init__(self, table: ir.Table, metadata: Metadata | None = None):
        self.table = table
        self.metadata = metadata or Metadata()

    def to_table(self) -> ir.Table:
        return self.table

    def to_pandas(self, categories: bool = False) -> pd.DataFrame:
        df = self.table.to_pandas()
        if categories:
            return self._categorize_pandas(df)
        return df

    def _categorize_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        for col, cats in self.metadata.categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            df[col] = pd.Categorical.from_codes(
                cast(Sequence[int], codes), cats.values, ordered=cats.ordered
            )
        return df

    def _categorize_pyarrow(self, table: pa.Table) -> pa.Table:
        if not self.metadata.categories:
            return table

        out = {}
        for name, col in zip(table.column_names, table.columns):
            cats = self.metadata.categories.get(name)
            if cats is not None:
                col = pa.chunked_array(
                    [
                        pa.DictionaryArray.from_arrays(
                            chunk, cats.values, ordered=cats.ordered
                        )
                        for chunk in col.chunks
                    ],
                )
            out[name] = col
        return pa.Table.from_pydict(out)

    def _categorize_pyarrow_batches(
        self, reader: pa.RecordBatchReader
    ) -> pa.RecordBatchReader:
        if not self.metadata.categories:
            return reader

        fields = []
        for field in reader.schema:
            cats = self.metadata.categories.get(field.name)
            if cats is not None:
                field = pa.field(
                    field.name,
                    pa.dictionary(field.type, cats.values.type, cats.ordered),
                    field.nullable,
                )
            fields.append(field)

        return pa.RecordBatchReader.from_batches(
            pa.schema(fields), _categorize_wrap_reader(reader, self.metadata.categories)
        )

    def to_pyarrow(self, categories: bool = False) -> pa.Table:
        table = self.table.to_pyarrow()
        if categories:
            table = self._categorize_pyarrow(table)
        return table

    def to_pyarrow_batches(self, categories: bool = False) -> pa.RecordBatchReader:
        reader = self.table.to_pyarrow_batches()
        if categories:
            return self._categorize_pyarrow_batches(reader)
        return reader


class Transform:
    def transform(self, table: ir.Table) -> ir.Table:
        raise NotImplementedError


class Step:
    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        raise NotImplementedError


class Recipe:
    steps: list[Step | Transform]

    def __init__(self, steps: Sequence[Step | Transform]):
        self.steps = list(steps)

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.steps)
        return f"Recipe:\n{parts}"

    def fit(
        self, table: ir.Table, outcomes: str | Sequence[str] | None = None
    ) -> RecipeTransform:
        if outcomes is None:
            outcomes = []
        elif isinstance(outcomes, str):
            outcomes = [outcomes]
        else:
            outcomes = list(outcomes)

        metadata = Metadata(outcomes)

        transforms = []
        for step in self.steps:
            if isinstance(step, Step):
                transform = step.fit(table, metadata)
            else:
                transform = step

            transforms.append(transform)
            table = transform.transform(table)

        return RecipeTransform(transforms, metadata)


class RecipeTransform:
    def __init__(self, transforms: Sequence[Transform], metadata: Metadata):
        self.transforms = list(transforms)
        self.metadata = metadata

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.transforms)
        return f"RecipeTransform:\n{parts}"

    def transform(self, table: ir.Table) -> Dataset:
        for transform in self.transforms:
            table = transform.transform(table)
        return Dataset(table, self.metadata)
