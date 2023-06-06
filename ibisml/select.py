from __future__ import annotations

import re
from collections.abc import Collection
from typing import Callable, Union

import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from ibisml.core import Metadata


class Selector:
    """The base selector class"""

    __slots__ = ()

    def __repr__(self):
        name = type(self).__name__
        args = ",".join(repr(getattr(self, n)) for n in self.__slots__)
        return f"{name}({args})"

    def __and__(self, other: SelectionType) -> Selector:
        selectors = []
        for part in [self, other]:
            if isinstance(part, and_):
                selectors.extend(part.selectors)
            else:
                selectors.append(part)
        return and_(*selectors)

    def __or__(self, other: SelectionType) -> Selector:
        selectors = []
        for part in [self, other]:
            if isinstance(part, or_):
                selectors.extend(part.selectors)
            else:
                selectors.append(part)
        return or_(*selectors)

    def __sub__(self, other: SelectionType) -> Selector:
        return self & ~selector(other)

    def __invert__(self) -> Selector:
        if isinstance(self, not_):
            return self.selector
        return not_(self)

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        """Whether the selector matches a given column"""
        raise NotImplementedError

    def select_columns(self, table: ir.Table, metadata: Metadata) -> list[str]:
        """Return a list of column names matching this selector."""
        return [
            c for c in table.columns if self.matches(table[c], metadata)  # type: ignore
        ]


SelectionType = Union[str, Collection[str], Callable[[ir.Column], bool], Selector]


def selector(obj: SelectionType) -> Selector:
    """Convert `obj` to a Selector"""
    if isinstance(obj, str):
        return cols(obj)
    elif isinstance(obj, Collection):
        return cols(*obj)
    elif callable(obj):
        return where(obj)
    elif isinstance(obj, Selector):
        return obj
    raise TypeError("Expected a str, list of strings, callable, or Selector")


class and_(Selector):
    __slots__ = ("selectors",)

    def __init__(self, *selectors):
        self.selectors = selectors

    def __repr__(self):
        args = " & ".join(repr(s) for s in self.selectors)
        return f"({args})"

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return all(s.matches(col, metadata) for s in self.selectors)


class or_(Selector):
    __slots__ = ("selectors",)

    def __init__(self, *selectors):
        self.selectors = selectors

    def __repr__(self):
        args = " | ".join(repr(s) for s in self.selectors)
        return f"({args})"

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return any(s.matches(col, metadata) for s in self.selectors)


class not_(Selector):
    __slots__ = ("selector",)

    def __init__(self, selector):
        self.selector = selector

    def __repr__(self):
        return f"~{self.selector!r}"

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return not self.selector.matches(col, metadata)


class everything(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return True


class cols(Selector):
    __slots__ = ("columns",)

    def __init__(self, columns: str | Collection[str]):
        if not isinstance(columns, str):
            columns = tuple(columns)
        self.columns = columns

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return col.get_name() in self.columns


class _StrMatcher(Selector):
    __slots__ = ("pattern",)

    def __init__(self, pattern: str):
        self.pattern = pattern


class contains(_StrMatcher):
    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return self.pattern in col.get_name()


class endswith(_StrMatcher):
    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return col.get_name().endswith(self.pattern)


class startswith(_StrMatcher):
    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return col.get_name().startswith(self.pattern)


class matches(_StrMatcher):
    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return re.search(self.pattern, col.get_name()) is not None


class has_type(Selector):
    __slots__ = ("type",)

    def __init__(self, type: dt.DataType | str | type[dt.DataType]):
        self.type = type

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        # A mapping of abstract or parametric types, to allow selecting all
        # subclasses/parametrizations of these types, rather than only a
        # specific instance.
        abstract = {
            "array": dt.Array,
            "decimal": dt.Decimal,
            "floating": dt.Floating,
            "geospatial": dt.GeoSpatial,
            "integer": dt.Integer,
            "map": dt.Map,
            "numeric": dt.Numeric,
            "struct": dt.Struct,
            "temporal": dt.Temporal,
        }
        if isinstance(self.type, str) and self.type.lower() in abstract:
            cls = abstract[self.type.lower()]
            return isinstance(col.type(), cls)
        elif isinstance(self.type, type):
            return isinstance(col.type(), self.type)
        else:
            return col.type() == dt.dtype(self.type)


class numeric(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        categories = metadata.get_categories(col.get_name())
        return isinstance(col.type(), dt.Numeric) and categories is None


class nominal(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return not isinstance(col.type(), dt.Numeric)


class categorical(Selector):
    __slots__ = ("ordered",)

    def __init__(self, ordered: bool | None = None):
        self.ordered = ordered

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        categories = metadata.get_categories(col.get_name())
        if categories is None:
            return False
        if self.ordered is not None:
            return categories.ordered == self.ordered
        return True


class where(Selector):
    __slots__ = ("predicate",)

    def __init__(self, predicate: Callable[[ir.Column], bool]):
        self.predicate = predicate

    def matches(self, col: ir.Column, metadata: Metadata) -> bool:
        return self.predicate(col)
