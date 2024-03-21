from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
import numpy as np
import pandas as pd
import pyarrow as pa
from ibis.common.dispatch import lazy_singledispatch

if TYPE_CHECKING:
    import dask.dataframe as dd
    import polars as pl
    import xgboost as xgb


def gen_name(prefix: str = "") -> str:
    """Create a unique identifier."""
    parts = ["ibis"]
    if prefix:
        parts.append(prefix)
    parts.append(os.urandom(8).hex())
    return "_".join(parts)


def _ibis_table_to_numpy(table: ir.Table) -> np.ndarray:
    if not all(t.is_numeric() for t in table.schema().types):
        raise ValueError(
            "Not all output columns are numeric, cannot convert to a numpy array"
        )
    return table.to_pandas().to_numpy()


def _y_as_dataframe(y: Any) -> pd.DataFrame:
    """Coerce `y` to a pandas dataframe"""
    if isinstance(y, pd.DataFrame):
        return y
    elif isinstance(y, pd.Series):
        return y.to_frame()
    y = np.asarray(y)
    if y.ndim == 1:
        return pd.DataFrame({"y": y})
    return pd.DataFrame(y, columns=[f"y{i}" for i in range(y.shape[-1])])


@lazy_singledispatch
def normalize_table(
    X: Any, y: Any = None, maintain_order: bool = False
) -> tuple[ir.Table, tuple[str, ...], str | None]:
    """Coerce `X` and `y` to an ibis table.

    Parameters
    ----------
    X : table-like
        The predictor columns in a supported input format.
    y : column-like or table-like, optional
        Any target columns, in a supported input format.
    maintain_order : bool, optional
        If True and `X` is an in-memory table, an index column will be inserted
        to use to maintain order of the output.

    Returns
    -------
    table : ir.Table
        The output table.
    targets : tuple[str, ...]
        A tuple of target column names in the table.
    index : str | None
        The index column name (if inserted), `None` otherwise.
    """
    raise TypeError(f"Cannot convert {type(X).__name__} to an ibis Table")


@normalize_table.register(ir.Table)
def _(X, y=None, maintain_order=False):
    if y is None:
        return X, (), None
    elif isinstance(y, (str, tuple, list)):
        targets = tuple(X.select(y).columns)
        return X, targets, None

    if isinstance(y, ir.Column):
        y = y.as_table()

    if not isinstance(y, ir.Table):
        raise TypeError(
            "When passing in `X` as an ibis Table, `y` must also be an "
            "ibis Table or Column"
        )

    if set(y.columns).intersection(X.columns):
        raise ValueError("X and y must not share column names")

    X_op = X.op()
    y_op = y.op()

    # For now we only handle reconstituting a table after a simple selection.
    # >>> X = parent[cols]
    # >>> y = parent[single_or_multiple_cols]
    # >>> table = parent[cols + single_or_multiple_cols]
    if (
        hasattr(ops, "Project")
        and isinstance(X_op, ops.Project)
        and isinstance(y_op, ops.Project)
        and X_op.parent is y_op.parent
    ):
        # ibis 9.0
        values = dict(X_op.values)
        values.update(y_op.values)
        table = ops.Project(X_op.parent, values).to_expr()
    elif (
        hasattr(ops, "Selection")
        and isinstance(X_op, ops.Selection)
        and isinstance(y_op, ops.Selection)
        and X_op.table is y_op.table
        and X_op.predicates == y_op.predicates
        and X_op.sort_keys == y_op.sort_keys
    ):
        # ibis 8.0
        table = ops.Selection(
            X_op.table,
            X_op.selections + y_op.selections,
            X_op.predicates,
            X_op.sort_keys,
        ).to_expr()
    else:
        raise ValueError("`X` and `y` must directly share a common parent table")

    return table, tuple(y.columns), None


@normalize_table.register(pd.DataFrame)
def _(X, y=None, maintain_order=False):
    if y is not None:
        y = _y_as_dataframe(y)
        table = pd.concat([X, y], axis=1)
        targets = tuple(y.columns)
    else:
        table = X
        targets = ()

    if maintain_order:
        index = gen_name("index")
        table = table.assign(**{index: np.arange(len(table))})
    else:
        index = None
    return ibis.memtable(table), targets, index


@normalize_table.register(np.ndarray)
def _(X, y=None, maintain_order=False):
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[-1])])
    if y is not None:
        y = _y_as_dataframe(y)
        table = pd.concat([X, y], axis=1)
        targets = tuple(y.columns)
    else:
        table = X
        targets = ()

    if maintain_order:
        index = gen_name("index")
        table = table.assign(**{index: np.arange(len(table))})
    else:
        index = None
    return ibis.memtable(table), targets, index


@normalize_table.register(pa.Table)
def _(X, y=None, maintain_order=False):
    if y is not None:
        if isinstance(y, (pa.ChunkedArray, pa.Array)):
            y = pa.Table.from_pydict({"y": y})
        elif not isinstance(y, pa.Table):
            raise TypeError(
                "When passing in `X` as a pyarrow.Table, `y` must also be a "
                "pyarrow.Table or Array"
            )
        targets = tuple(y.column_names)
        table = X
        for name in y.column_names:
            table = table.append_column(name, y[name])
    else:
        table = X
        targets = ()

    if maintain_order:
        index = gen_name("index")
        table = table.append_column(index, pa.array(np.arange(len(table))))
    else:
        index = None
    return ibis.memtable(table), targets, index


@normalize_table.register("polars.DataFrame")
def _(X, y=None, maintain_order=False):
    import polars as pl

    if y is not None:
        if isinstance(y, pl.Series):
            y = y.to_frame()
        elif not isinstance(y, pl.DataFrame):
            raise TypeError(
                "When passing in `X` as a polars.DataFrame, `y` must also be "
                "a polars.DataFrame or Series"
            )
        table = pl.concat([X, y], how="horizontal")
        targets = tuple(y.columns)
    else:
        table = X
        targets = ()

    if maintain_order:
        index = gen_name("index")
        table = table.with_columns(**{index: pl.arange(len(table))})
    else:
        index = None
    return ibis.memtable(table), targets, index


class Metadata:
    def __init__(
        self,
        categories: dict[str, pa.Array] | None = None,
        targets: tuple[str, ...] = (),
    ):
        self.categories = categories or {}
        self.targets = targets

    def get_categories(self, column: str) -> pa.Array | None:
        return self.categories.get(column)

    def set_categories(self, column: str, values: pa.Array | list[Any]) -> None:
        self.categories[column] = pa.array(values)

    def drop_categories(self, column: str) -> None:
        self.categories.pop(column, None)


def _categorize_wrap_reader(
    reader: pa.RecordBatchReader, categories: dict[str, pa.Array]
) -> Iterable[pa.RecordBatch]:
    for batch in reader:
        out = {}
        for name, col in zip(batch.schema.names, batch.columns):
            cats = categories.get(name)
            if cats is not None:
                col = pa.DictionaryArray.from_arrays(col, cats)
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

    def _fit_table(
        self, table: ir.Table, targets: tuple[str, ...] = (), index: str | None = None
    ) -> None:
        metadata = Metadata(targets=targets)

        if index is not None:
            table = table.drop(index)

        for step in self.steps:
            step.fit_table(table, metadata)
            table = step.transform_table(table)

        self.metadata_ = metadata

    def _transform_table(
        self, table: ir.Table, targets: tuple[str, ...] = (), index: str | None = None
    ) -> ir.Table:
        if targets:
            table = table.drop(*targets)

        for step in self.steps:
            table = step.transform_table(table)

        if index is not None:
            table = table.order_by(index).drop(index)

        return table

    def _to_output_format(self, table: ir.Table) -> Any:
        if self._output_format == "pandas":
            return table.to_pandas()
        elif self._output_format == "polars":
            return table.to_polars()
        elif self._output_format == "pyarrow":
            return table.to_pyarrow()
        else:
            assert self._output_format == "default"
            return _ibis_table_to_numpy(table)

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
        table, targets, index = normalize_table(X, y)
        self._fit_table(table, targets, index)
        return self

    def to_ibis(self, X) -> ir.Table:
        """Transform X and return an ibis table.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        table, targets, index = normalize_table(X, maintain_order=True)
        return self._transform_table(table, targets, index)

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
        table = self.to_ibis(X)
        return self._to_output_format(table)

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
        table, targets, index = normalize_table(X, y, maintain_order=True)
        self._fit_table(table, targets, index)
        table = self._transform_table(table, targets, index)
        return self._to_output_format(table)

    def _categorize_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        for col, cats in self.metadata_.categories.items():
            codes = df[col].fillna(-1)
            if not pd.api.types.is_integer_dtype(codes):
                codes = codes.astype("int64")
            df[col] = pd.Categorical.from_codes(cast(Sequence[int], codes), cats)
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
                        pa.DictionaryArray.from_arrays(chunk, cats)
                        for chunk in col.chunks
                    ]
                )
            out[name] = col
        return pa.Table.from_pydict(out)

    def _categorize_dask_dataframe(self, ddf: dd.DataFrame) -> dd.DataFrame:
        if not self.metadata_.categories:
            return ddf
        categorize = _get_categorize_chunk()
        return ddf.map_partitions(categorize, self.metadata_.categories)

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
                    field.name, pa.dictionary(field.type, cats.type), field.nullable
                )
            fields.append(field)

        return pa.RecordBatchReader.from_batches(
            pa.schema(fields),
            _categorize_wrap_reader(reader, self.metadata_.categories),
        )

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
        df = self.to_ibis(X).to_pandas()
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
        table = self.to_ibis(X)
        return _ibis_table_to_numpy(table)

    def to_polars(self, X: Any) -> pl.DataFrame:
        """Transform X and return a ``polars.DataFrame``.

        Parameters
        ----------
        X : table-like
            The input data to transform.
        """
        return self.to_ibis(X).to_polars()

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
        table = self.to_ibis(X).to_pyarrow()
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
        reader = self.to_ibis(X).to_pyarrow_batches()
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

        table = self.to_ibis(X)

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
