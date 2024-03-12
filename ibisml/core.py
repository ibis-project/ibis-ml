from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import ibis
import ibis.expr.types as ir
import numpy as np
import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    import dask.dataframe as dd
    import polars as pl
    import xgboost as xgb


def _as_table(X: Any):
    if isinstance(X, ir.Table):
        return X
    elif isinstance(X, np.ndarray):
        return ibis.memtable(
            pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[-1])])
        )
    else:
        return ibis.memtable(X)


class Categories:
    def __init__(self, values: pa.Array):
        self.values = values

    def __repr__(self):
        items = [repr(v) for v in self.values[:3].to_pylist()]
        if len(self.values) > 3:
            items.append("...")
        values = ", ".join(items)
        return f"Categories<{values}>"


class Metadata:
    def __init__(self, categories: dict[str, Categories] | None = None):
        self.categories = categories or {}

    def get_categories(self, column: str) -> Categories | None:
        return self.categories.get(column)

    def set_categories(self, column: str, values: pa.Array | list[Any]) -> None:
        self.categories[column] = Categories(pa.array(values))

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
                col = pa.DictionaryArray.from_arrays(col, cats.values)
            out[name] = col
        yield pa.RecordBatch.from_pydict(out)


@cache
def _get_categorize_chunk() -> Callable[[str, list[str], Any], pd.DataFrame]:
    """Wrap the `categorize` function in a closure, so cloudpickle will encode
    the full function.

    This avoids requiring `ibisml` or `ibis` exist on the worker nodes of the
    dask cluster.
    """

    def categorize(df: pd.DataFrame, categories: dict[str, list[Any]]) -> pd.DataFrame:
        import pandas as pd

        new = {}
        for col, cats in categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            new[col] = pd.Categorical.from_codes(codes, cats)

        return df.assign(**new)

    return categorize


class Step:
    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        pass

    def transform_table(self, table: ir.Table) -> ir.Table:
        pass

    def is_fitted(self) -> bool:
        """Check if a step has already been fit."""
        return any(n.endswith("_") and not n.endswith("__") for n in dir(self))

    def clone(self) -> Step:
        """Return an unfit copy of this Step."""
        out = copy.copy(self)
        for n in dir(self):
            if n.endswith("_") and not n.endswith("__"):
                delattr(out, n)
        return out


class Recipe:
    def __init__(self, *steps: Step):
        self.steps = steps
        self._output_format = "default"

    @property
    def output_format(self) -> Literal["default", "pandas", "pyarrow", "polars"]:
        """The output format to use for ``transform``"""
        return self._output_format

    def get_params(self, deep=True):
        return {"steps": self.steps}

    def set_params(self, **kwargs):
        if "steps" in kwargs:
            self.steps = kwargs.get("steps")

    def set_output(
        self,
        *,
        transform: Literal["default", "pandas", "pyarrow", "polars", None] = None,
    ) -> Recipe:
        """Set output type returned by `transform`.

        This is part of the standard Scikit-Learn API.

        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: Pandas dataframe
            - `"polars"`: Polars dataframe
            - `"pyarrow"`: Pyarrow table
            - `None`: Transform configuration is unchanged

        """
        if transform is None:
            return self

        formats = ("default", "pandas", "polars", "pyarrow")

        if transform not in formats:
            raise ValueError(
                f"`transform` must be one of {formats!r}, got {transform!r}"
            )

        self._output_format = transform
        return self

    def __sklearn_clone__(self) -> Recipe:
        steps = [s.clone() for s in self.steps]
        return Recipe(*steps)

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted()

    def is_fitted(self) -> bool:
        """Check if this recipe has already been fit."""
        return all(s.is_fitted() for s in self.steps)

    def fit(self, X, y=None) -> Recipe:
        """Fit a recipe.

        Parameters
        ----------
        X : table-like
            Training data.
        y : column-like, optional
            Training targets.

        Returns
        -------
        self
            Returns the same instance.
        """
        table = _as_table(X)
        metadata = Metadata()
        for step in self.steps:
            step.fit_table(table, metadata)
            table = step.transform_table(table)
        self.metadata_ = metadata
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : table-like
            Data to transform.

        Returns
        -------
        Xt
            Transformed data.
        """
        if self._output_format == "pandas":
            return self.to_pandas(X)
        elif self._output_format == "polars":
            return self.to_polars(X)
        elif self._output_format == "pyarrow":
            return self.to_pyarrow(X)
        else:
            assert self._output_format == "default"
            return self.to_numpy(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Parameters
        ----------
        X : table-like
            Training data.
        y : column-like, optional
            Training targets.

        Returns
        -------
        Xt
            Transformed training data.
        """
        return self.fit(X, y).transform(X)

    def _categorize_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        for col, cats in self.metadata_.categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            df[col] = pd.Categorical.from_codes(cast(Sequence[int], codes), cats.values)
        return df

    def _categorize_pyarrow(self, table: pa.Table) -> pa.Table:
        if not self.metadata_.categories:
            return table

        out = {}
        for name, col in zip(table.column_names, table.columns):
            cats = self.metadata_.categories.get(name)
            if cats is not None:
                col = pa.chunked_array(
                    [
                        pa.DictionaryArray.from_arrays(chunk, cats.values)
                        for chunk in col.chunks
                    ],
                )
            out[name] = col
        return pa.Table.from_pydict(out)

    def _categorize_dask_dataframe(self, ddf: dd.DataFrame) -> dd.DataFrame:
        if not self.metadata_.categories:
            return ddf

        categorize = _get_categorize_chunk()

        categories = {
            col: cats.values for col, cats in self.metadata_.categories.items()
        }
        return ddf.map_partitions(categorize, categories)

    def _categorize_pyarrow_batches(
        self, reader: pa.RecordBatchReader
    ) -> pa.RecordBatchReader:
        if not self.metadata_.categories:
            return reader

        fields = []
        for field in reader.schema:
            cats = self.metadata_.categories.get(field.name)
            if cats is not None:
                field = pa.field(
                    field.name,
                    pa.dictionary(field.type, cats.values.type),
                    field.nullable,
                )
            fields.append(field)

        return pa.RecordBatchReader.from_batches(
            pa.schema(fields),
            _categorize_wrap_reader(reader, self.metadata_.categories),
        )

    def to_table(self, X: ir.Table) -> ir.Table:
        """Transform X and return an ibis table.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        table = _as_table(X)
        for step in self.steps:
            table = step.transform_table(table)
        return table

    def to_pandas(self, X: Any, categories: bool = False) -> pd.DataFrame:
        """Transform X and return a ``pandas.DataFrame``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        categories : bool
            Whether to return any categorical columns as pandas categorical
            series. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        df = self.to_table(X).to_pandas()
        if categories:
            return self._categorize_pandas(df)
        return df

    def to_numpy(self, X: Any) -> np.ndarray:
        """Transform X and return a ``numpy.ndarray``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        table = self.to_table(X)
        if not all(t.is_numeric() for t in table.schema().types):
            raise ValueError(
                "Not all output columns are numeric, cannot convert to a numpy array"
            )
        return table.to_pandas().values

    def to_polars(self, X: Any) -> pl.DataFrame:
        """Transform X and return a ``polars.DataFrame``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        return self.to_table(X).to_polars()

    def to_pyarrow(self, X: Any, categories: bool = False) -> pa.Table:
        """Transform X and return a ``pyarrow.Table``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        categories : bool
            Whether to return any categorical columns as dictionary-encoded
            columns. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        table = self.to_table(X).to_pyarrow()
        if categories:
            table = self._categorize_pyarrow(table)
        return table

    def to_pyarrow_batches(
        self, X: Any, categories: bool = False
    ) -> pa.RecordBatchReader:
        """Transform X and return a ``pyarrow.RecordBatchReader``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        categories : bool
            Whether to return any categorical columns as dictionary-encoded
            columns. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        reader = self.to_table(X).to_pyarrow_batches()
        if categories:
            return self._categorize_pyarrow_batches(reader)
        return reader

    def to_dask_dataframe(self, X: Any, categories: bool = False) -> dd.DataFrame:
        """Transform X and return a ``dask.dataframe.DataFrame``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        categories : bool
            Whether to return any categorical columns as dask categorical
            series. If False (the default) these columns will be returned
            as numeric columns containing only their integral categorical
            codes.
        """
        import dask.dataframe as dd

        table = self.to_table(X)

        con = ibis.get_backend(table)
        if hasattr(con, "to_dask"):
            ddf = con.to_dask(table)
            if categories:
                return self._categorize_dask_dataframe(ddf)
            return ddf
        else:
            # TODO(jcrist): this is suboptimal, but may not matter. In practice I'd only
            # expect the dask conversion path to be used for backends where dask
            # integration makes sense.
            df = table.to_pandas()
            if categories:
                table = self._categorize_pandas(df)
            return dd.from_pandas(df, npartitions=1)

    def to_dmatrix(self, X: Any) -> xgb.DMatrix:
        """Transform X and return a ``xgboost.DMatrix``

        Parameters
        ----------
        X : table-like
            The input data to transform.

        """
        import xgboost as xgb

        df = self.to_pandas(X, categories=True)
        return xgb.DMatrix(
            df[self.features], df[self.outcomes], enable_categorical=True
        )

    def to_dask_dmatrix(self, X: Any) -> xgb.dask.DaskDMatrix:
        """Transform X and return a ``xgboost.dask.DMatrix``

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        import xgboost as xgb
        from dask.distributed import get_client

        ddf = self.to_dask_dataframe(X, categories=True)
        return xgb.dask.DaskDMatrix(
            get_client(),
            ddf[self.features],
            ddf[self.outcomes],
            enable_categorical=True,
        )
