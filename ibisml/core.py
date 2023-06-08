from __future__ import annotations

from collections.abc import Sequence, Iterable
from typing import Any, cast, TYPE_CHECKING

import pyarrow as pa
import ibis
import ibis.expr.types as ir

if TYPE_CHECKING:
    import pandas as pd


class Categories:
    def __init__(self, values: pa.Array, ordered: bool = False):
        self.values = values
        self.ordered = ordered


class Metadata:
    def __init__(
        self,
        outcomes: list[str] | None = None,
        categories: dict[str, Categories] | None = None,
    ):
        self.outcomes = outcomes or []
        self.categories = categories or {}

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


class TransformedTable:
    def __init__(
        self,
        table: ir.Table,
        features: list[str] | None = None,
        outcomes: list[str] | None = None,
        categories: dict[str, Categories] | None = None,
    ):
        if outcomes is None:
            outcomes = []
        if features is None:
            features = [c for c in table.columns if c not in outcomes]
        if categories is None:
            categories = {}

        self.table = table
        self.features = features
        self.outcomes = outcomes
        self.categories = categories

    @property
    def schema(self) -> ibis.Schema:
        return self.table.schema()

    def to_table(self) -> ir.Table:
        return self.table

    def to_pandas(self, categories: bool = False) -> pd.DataFrame:
        df = self.table.to_pandas()
        if categories:
            return self._categorize_pandas(df)
        return df

    def _categorize_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        for col, cats in self.categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            df[col] = pd.Categorical.from_codes(
                cast(Sequence[int], codes), cats.values, ordered=cats.ordered
            )
        return df

    def _categorize_pyarrow(self, table: pa.Table) -> pa.Table:
        if not self.categories:
            return table

        out = {}
        for name, col in zip(table.column_names, table.columns):
            cats = self.categories.get(name)
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
        if not self.categories:
            return reader

        fields = []
        for field in reader.schema:
            cats = self.categories.get(field.name)
            if cats is not None:
                field = pa.field(
                    field.name,
                    pa.dictionary(field.type, cats.values.type, cats.ordered),
                    field.nullable,
                )
            fields.append(field)

        return pa.RecordBatchReader.from_batches(
            pa.schema(fields), _categorize_wrap_reader(reader, self.categories)
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
    def __repr__(self) -> str:
        cls_name = type(self).__name__
        parts = []

        def quote(col: str):
            return repr(col) if " " in col else col

        if len(self.input_columns) <= 4:
            parts = [quote(c) for c in self.input_columns]
        else:
            parts = [quote(c) for c in self.input_columns[:2]]
            parts.append("...")
            parts.extend(quote(c) for c in self.input_columns[-2:])
        return f"{cls_name}<{', '.join(parts)}>"

    @property
    def input_columns(self) -> list[str]:
        raise NotImplementedError

    def transform(self, table: ir.Table) -> ir.Table:
        raise NotImplementedError


class Step:
    def _repr(self, *args: str, **kwargs: Any) -> str:
        parts = [repr(getattr(self, name)) for name in args]
        for name, default in kwargs.items():
            value = getattr(self, name)
            if value != default:
                parts.append(f"{name}={value!r}")
        cls_name = type(self).__name__
        return f"{cls_name}({', '.join(parts)})"

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

        input_schema = table.drop(*outcomes).schema()

        metadata = Metadata(outcomes)

        transforms = []
        for step in self.steps:
            if isinstance(step, Step):
                transform = step.fit(table, metadata)
            else:
                transform = step

            transforms.append(transform)
            table = transform.transform(table)

        output_schema = table.drop(*outcomes).schema()

        return RecipeTransform(
            transforms,
            metadata=metadata,
            input_schema=input_schema,
            output_schema=output_schema,
        )


class RecipeTransform:
    def __init__(
        self,
        transforms: Sequence[Transform],
        metadata: Metadata,
        input_schema: ibis.Schema,
        output_schema: ibis.Schema,
    ):
        self.transforms = list(transforms)
        self.metadata = metadata
        self.input_schema = input_schema
        self.output_schema = output_schema

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.transforms)
        return f"RecipeTransform:\n{parts}"

    def __call__(self, table: ir.Table) -> TransformedTable:
        return self.transform(table)

    def transform(self, table: ir.Table) -> TransformedTable:
        if table.schema() != self.input_schema:
            # Schemas don't match, cast, erroring if not possible
            table = table.cast(self.input_schema)

        for transform in self.transforms:
            table = transform.transform(table)

        features = list(self.output_schema.names)
        outcomes = [col for col in self.metadata.outcomes if col in table.columns]
        seen = set(features).union(outcomes)
        extra = [col for col in table.columns if col not in seen]
        ordered_cols = [*features, *outcomes, *extra]

        # Reorder the columns to a consistent order
        if table.columns != ordered_cols:
            table = table.select(ordered_cols)

        return TransformedTable(
            table,
            features,
            outcomes,
            self.metadata.categories,
        )
