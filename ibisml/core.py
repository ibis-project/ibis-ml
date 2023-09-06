from __future__ import annotations

from collections.abc import Sequence, Iterable
from typing import Any, Callable, cast, TYPE_CHECKING
from functools import cache

import pyarrow as pa
import ibis
import ibis.expr.types as ir

if TYPE_CHECKING:
    import pandas as pd
    import dask.dataframe as dd
    import xgboost as xgb


class Categories:
    def __init__(self, values: pa.Array, ordered: bool = False):
        self.values = values
        self.ordered = ordered

    def __repr__(self):
        items = [repr(v) for v in self.values[:3].to_pylist()]
        if len(self.values) > 3:
            items.append("...")
        items.append(f"ordered={self.ordered}")
        values = ", ".join(items)
        return f"Categories<{values}>"


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


@cache
def _get_categorize_chunk() -> Callable[[str, list[str], Any], pd.DataFrame]:
    """Wrap the `categorize` function in a closure, so cloudpickle will encode
    the full function.

    This avoids requiring `ibisml` or `ibis` exist on the worker nodes of the
    dask cluster.
    """

    def categorize(
        df: pd.DataFrame,
        categories: dict[str, tuple[list[Any], bool]],
    ) -> pd.DataFrame:
        import pandas as pd

        new = {}
        for col, (cats, ordered) in categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            new[col] = pd.Categorical.from_codes(codes, cats, ordered=ordered)

        return df.assign(**new)

    return categorize


class TransformResult:
    """The result of applying a RecipeTransform.

    This is a wrapper around an ibis Table containing an expression describing
    the full preprocessing pipeline. Use one of the `to_*` methods to execute
    the expression and convert the result into an input type usable for fitting
    your model.
    """

    def __init__(
        self,
        table: ir.Table,
        features: list[str] | None = None,
        outcomes: list[str] | None = None,
        other: list[str] | None = None,
        categories: dict[str, Categories] | None = None,
    ):
        self.table = table
        self.features = features or []
        self.outcomes = outcomes or []
        self.other = other or []
        self.categories = categories or {}

    def __repr__(self):
        schema = self.schema
        parts = ["TransformResult:"]
        for group in ("features", "outcomes", "other"):
            columns = getattr(self, group)
            if not columns:
                continue
            width = max(len(c) for c in columns)
            rows = "".join(
                f"    {col.ljust(width)}  {schema[col]}\n" for col in columns
            )
            parts.append(f"- {group.capitalize()} {{\n{rows}}}")

        return "\n".join(parts)

    @property
    def schema(self) -> ibis.Schema:
        """The transformed table schema"""
        return self.table.schema()

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

    def _categorize_dask_dataframe(self, ddf: dd.DataFrame) -> dd.DataFrame:
        if not self.categories:
            return ddf

        categorize = _get_categorize_chunk()

        categories = {
            col: (cats.values, cats.ordered) for col, cats in self.categories.items()
        }
        return ddf.map_partitions(categorize, categories)

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

    def to_table(self) -> ir.Table:
        """Convert to an ibis Table"""
        return self.table

    def to_pandas(self, categories: bool = False) -> pd.DataFrame:
        """Convert to a ``pandas.DataFrame``.

        Parameters
        ----------
        categories
            Whether to return any categorical columns as pandas categorical
            series. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        df = self.table.to_pandas()
        if categories:
            return self._categorize_pandas(df)
        return df

    def to_pyarrow(self, categories: bool = False) -> pa.Table:
        """Convert to a ``pyarrow.Table``.

        Parameters
        ----------
        categories
            Whether to return any categorical columns as dictionary-encoded
            columns. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        table = self.table.to_pyarrow()
        if categories:
            table = self._categorize_pyarrow(table)
        return table

    def to_pyarrow_batches(self, categories: bool = False) -> pa.RecordBatchReader:
        """Convert to a ``pyarrow.RecordBatchReader``.

        Parameters
        ----------
        categories
            Whether to return any categorical columns as dictionary-encoded
            columns. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        reader = self.table.to_pyarrow_batches()
        if categories:
            return self._categorize_pyarrow_batches(reader)
        return reader

    def to_dask_dataframe(self, categories: bool = False) -> dd.DataFrame:
        """Convert to a ``dask.dataframe.DataFrame``.

        Parameters
        ----------
        categories
            Whether to return any categorical columns as dask categorical
            series. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        import dask.dataframe as dd

        con = ibis.get_backend(self.table)
        if hasattr(con, "to_dask"):
            ddf = con.to_dask(self.table)
            if categories:
                return self._categorize_dask_dataframe(ddf)
            return ddf
        else:
            # TODO: this is suboptimal, but may not matter. In practice I'd only
            # expect the dask conversion path to be used for backends where dask
            # integration makes sense.
            return dd.from_pandas(self.to_pandas(categories=categories), npartitions=1)

    def to_dmatrix(self) -> xgb.DMatrix:
        """Convert to a ``xgboost.DMatrix``"""
        import xgboost as xgb

        df = self.to_pandas(categories=True)
        return xgb.DMatrix(
            df[self.features], df[self.outcomes], enable_categorical=True
        )

    def to_dask_dmatrix(self) -> xgb.dask.DaskDMatrix:
        """Convert to a ``xgboost.dask.DMatrix``"""
        import xgboost as xgb
        from dask.distributed import get_client

        ddf = self.to_dask_dataframe(categories=True)
        return xgb.dask.DaskDMatrix(
            get_client(),
            ddf[self.features],
            ddf[self.outcomes],
            enable_categorical=True,
        )


class Transform:
    """The base Transform."""

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
    """The base Step."""

    def _repr(self) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        raise NotImplementedError

    def __repr__(self) -> str:
        parts = []
        for name, value in self._repr():
            if name:
                parts.append(f"{name}={value!r}")
            else:
                parts.append(repr(value))
        cls_name = type(self).__name__
        return f"{cls_name}({', '.join(parts)})"


class Recipe:
    """A recipe for fitting a preprocessing transform.

    Recipes combine one or more preprocessing steps, applied in series.

    Parameters
    ----------
    *steps
        One or more preprocessing steps.
    """

    steps: list[Step | Transform]

    def __init__(self, *steps: Step | Transform):
        self.steps = list(steps)

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.steps)
        return f"Recipe:\n{parts}"

    def fit(
        self, table: ir.Table, outcomes: str | Sequence[str] | None = None
    ) -> RecipeTransform:
        """Fit the recipe.

        Parameters
        ----------
        table
            The training data
        outcomes
            The column (or columns) to use as outcome variables.

        Returns
        -------
        RecipeTransform
        """

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
    """The result of fitting a Recipe."""

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

    def __call__(self, table: ir.Table) -> TransformResult:
        """Apply the transform to new data."""
        return self.transform(table)

    def transform(self, table: ir.Table) -> TransformResult:
        """Apply the transform to new data.

        Parameters
        ----------
        table
            An ibis table.

        Returns
        -------
        TransformResult
        """
        missing = set(self.input_schema).difference(table.columns)
        if missing:
            formatted_cols = "\n".join(f"- {c!r}" for c in sorted(missing))
            raise ValueError(f"Missing required columns:\n{formatted_cols}")
        if table.schema() != self.input_schema:
            # Schemas don't match, cast, erroring if not possible
            table = table.cast(self.input_schema)

        for transform in self.transforms:
            table = transform.transform(table)

        features = list(self.output_schema.names)
        outcomes = [col for col in self.metadata.outcomes if col in table.columns]
        seen = set(features).union(outcomes)
        other = [col for col in table.columns if col not in seen]
        ordered_cols = [*features, *outcomes, *other]

        # Reorder the columns to a consistent order
        if table.columns != ordered_cols:
            table = table.select(ordered_cols)

        return TransformResult(
            table,
            features,
            outcomes,
            other,
            self.metadata.categories,
        )
