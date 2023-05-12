import re
from collections.abc import Sequence
from typing import Callable

import ibis.expr.types as ir
import ibis.expr.datatypes as dt


class Selector:
    """The base selector class"""

    __slots__ = ()

    def __repr__(self):
        name = type(self).__name__
        args = ",".join(repr(getattr(self, n)) for n in self.__slots__)
        return f"{name}({args})"

    def __and__(self, other):
        selectors = []
        for part in [self, other]:
            if isinstance(part, and_):
                selectors.extend(part.selectors)
            else:
                selectors.append(part)
        return and_(*selectors)

    def __or__(self, other):
        selectors = []
        for part in [self, other]:
            if isinstance(part, or_):
                selectors.extend(part.selectors)
            else:
                selectors.append(part)
        return or_(*selectors)

    def __invert__(self):
        if isinstance(self, not_):
            return self.selector
        return not_(self)

    def matches(self, col: ir.Column) -> bool:
        """Whether the selector matches a given column"""
        raise NotImplementedError


class and_(Selector):
    __slots__ = ("selectors",)

    def __init__(self, *selectors):
        self.selectors = selectors

    def __repr__(self):
        args = " & ".join(repr(s) for s in self.selectors)
        return f"({args})"

    def matches(self, col: ir.Column) -> bool:
        return all(s.matches(col) for s in self.selectors)


class or_(Selector):
    __slots__ = ("selectors",)

    def __init__(self, *selectors):
        self.selectors = selectors

    def __repr__(self):
        args = " | ".join(repr(s) for s in self.selectors)
        return f"({args})"

    def matches(self, col: ir.Column) -> bool:
        return any(s.matches(col) for s in self.selectors)


class not_(Selector):
    __slots__ = ("selector",)

    def __init__(self, selector):
        self.selector = selector

    def __repr__(self):
        return f"~{self.selector!r}"

    def matches(self, col: ir.Column) -> bool:
        return not self.selector.matches(col)


class all(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column) -> bool:
        return True


class cols(Selector):
    __slots__ = ("columns",)

    def __init__(self, columns: str | Sequence[str]):
        if not isinstance(columns, str):
            columns = tuple(columns)
        self.columns = columns

    def matches(self, col: ir.Column) -> bool:
        return col in self.columns


class _StrMatcher(Selector):
    __slots__ = ("pattern",)

    def __init__(self, pattern: str):
        self.pattern = pattern


class contains(_StrMatcher):
    def matches(self, col: ir.Column) -> bool:
        return self.pattern in col.name


class endswith(_StrMatcher):
    def matches(self, col: ir.Column) -> bool:
        return col.name.endswith(self.pattern)


class startswith(_StrMatcher):
    def matches(self, col: ir.Column) -> bool:
        return col.name.startswith(self.pattern)


class matches(_StrMatcher):
    def matches(self, col: ir.Column) -> bool:
        return re.search(self.pattern, col.name) is not None


class of_type(Selector):
    __slots__ = ("type",)

    def __init__(self, type: dt.DataType | str | type[dt.DataType]):
        self.type = type

    def matches(self, col: ir.Column) -> bool:
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
            cls = abstract.get(self.type.lower())
            return isinstance(col.type(), cls)
        elif isinstance(self.type, type):
            return isinstance(col.type(), self.type)
        else:
            return col.type() == dt.dtype(self.type)


class numeric(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column) -> bool:
        return isinstance(col.type(), dt.Numeric)


class nominal(Selector):
    __slots__ = ()

    def matches(self, col: ir.Column) -> bool:
        return not isinstance(col.type(), dt.Numeric)


class where(Selector):
    __slots__ = ("predicate",)

    def __init__(self, predicate: Callable[[ir.Column], bool]):
        self.predicate = predicate

    def matches(self, col: ir.Column) -> bool:
        return self.predicate(col)
