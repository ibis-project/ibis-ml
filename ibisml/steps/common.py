from __future__ import annotations

from typing import Callable, Iterable, Any

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.expr.deferred import Deferred

import ibisml as ml
from ibisml.core import Metadata, Step, Transform
from ibisml.select import SelectionType, selector


class Drop(Step):
    """A step for dropping selected columns from the output.

    Parameters
    ----------
    inputs
        A selection of columns to drop.

    Examples
    --------
    >>> import ibisml as ml

    Drop all non-numeric columns

    >>> step = ml.Drop(~ml.numeric())

    Drop specific columns by name

    >>> step = ml.Drop(["x", "y"])
    """

    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Drop(columns)


class Cast(Step):
    """A step for casting selected columns to a specific dtype.

    Parameters
    ----------
    inputs
        A selection of columns to cast.
    dtype
        The dtype to cast to. May be a dtype instance, class, or a string
        representation of one.

    Examples
    --------
    >>> import ibisml as ml

    Cast all numeric columns to float64

    >>> step = ml.Cast(ml.numeric(), "float64")

    Cast specific columns to int64 by name

    >>> step = ml.Cast(["x", "y"], "int64")
    """

    def __init__(
        self,
        inputs: SelectionType,
        dtype: dt.DataType | type[dt.DataType] | str,
    ):
        self.inputs = selector(inputs)
        self.dtype = dt.dtype(dtype)

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("", str(self.dtype))

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.Cast(columns, self.dtype)


class MutateAt(Step):
    """A step for mutating a selection of columns.

    Parameters
    ----------
    inputs
        A selection of columns to use as inputs to ``expr``/``named_exprs``.
    expr
        An optional callable (``Column -> Column``) or deferred expression to
        apply to all columns in inputs. Output columns will have the same name
        as their respective inputs (effectively replacing them in the output
        table).
    named_exprs
        Named callables (``Column -> Column``) or deferred expressions to apply
        to all columns in inputs. Output columns will be named
        ``{column}_{name}`` where ``column`` is the input column name and
        ``name`` is the expression/callable name.

    Examples
    --------
    >>> import ibisml as ml
    >>> from ibis import _

    Replace all numeric columns with their absolute values.

    >>> step = ml.MutateAt(ml.numeric(), _.abs())

    Same as the above, but instead create new columns with ``_abs`` suffixes.

    >>> step = ml.MutateAt(ml.numeric(), abs=_.abs())
    """

    def __init__(
        self,
        inputs: SelectionType,
        expr: Callable[[ir.Column], ir.Column] | Deferred | None = None,
        /,
        **named_exprs: Callable[[ir.Column], ir.Column] | Deferred,
    ):
        self.inputs = selector(inputs)
        self.expr = expr
        self.named_exprs = named_exprs

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        if self.expr is not None:
            yield ("", self.expr)
        for name, expr in self.named_exprs.items():
            yield name, expr

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        columns = self.inputs.select_columns(table, metadata)
        return ml.transforms.MutateAt(
            columns, expr=self.expr, named_exprs=self.named_exprs
        )


class Mutate(Step):
    """A step for defining new columns with Ibis.

    Parameters
    ----------
    exprs
        Callables (``Table -> Column``) or deferred expressions to use to define
        new columns in the output table.
    named_exprs
        Named callables (``Table -> Column``) or deferred expressions to use to
        define new columns in the output table.

    Examples
    --------
    >>> import ibisml as ml
    >>> from ibis import _

    Define a new column ``c`` as ``a**2 + b**2``

    >>> step = ml.Mutate(c=_.a**2 + _.b**2)
    """

    def __init__(
        self,
        *exprs: Callable[[ir.Table], ir.Column] | Deferred,
        **named_exprs: Callable[[ir.Table], ir.Column] | Deferred,
    ):
        self.exprs = exprs
        self.named_exprs = named_exprs

    def _repr(self) -> Iterable[tuple[str, Any]]:
        for expr in self.exprs:
            yield "", expr
        for name, expr in self.named_exprs.items():
            yield name, expr

    def fit(self, table: ir.Table, metadata: Metadata) -> Transform:
        return ml.transforms.Mutate(*self.exprs, **self.named_exprs)
