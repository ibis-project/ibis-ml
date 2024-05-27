import pprint

from ibis_ml.core import Recipe, Step


def _pprint_recipe(self, object, stream, indent, allowance, context, level):
    stream.write(object.__class__.__name__ + "(")
    if getattr(self, "_indent_at_name", True):
        indent += len(object.__class__.__name__)

    self._format_items(object.steps, stream, indent, allowance + 1, context, level)
    stream.write(")")


def _pprint_step(self, object, stream, indent, allowance, context, level):
    stream.write(object.__class__.__name__ + "(")
    if getattr(self, "_indent_at_name", True):
        indent += len(object.__class__.__name__)

    indent += self._indent_per_level
    delimnl = ",\n" + " " * indent
    delim = ""
    it = object._repr()  # noqa: SLF001
    try:
        next_ent = next(it)
    except StopIteration:
        return
    last = False
    n_items = 0
    while not last:
        if n_items == getattr(self, "n_max_elements_to_show", None):
            stream.write(", ...")
            break
        n_items += 1
        ent = next_ent
        try:
            next_ent = next(it)
        except StopIteration:
            last = True
        # TODO(deepyaman): Support `compact` option for `PrettyPrinter`.
        stream.write(delim)
        delim = delimnl
        k, v = ent
        if k:
            rep = self._repr(k, context, level)
            rep = rep.strip("'")
            middle = "="
            stream.write(rep)
            stream.write(middle)
        else:
            rep = middle = ""
        self._format(
            v,
            stream,
            indent + len(rep) + len(middle),
            allowance + 1 if last else 1,
            context,
            level,
        )

    stream.write(")")


def _safe_repr(self, object, context, maxlevels, level):
    """Return triple (repr_string, isreadable, isrecursive).

    Notes
    -----
    Same as the built-in ``pprint.PrettyPrinter._safe_repr``, with added
    support for ``Recipe`` and ``Step`` objects.

    References
    ----------
    .. [1] https://github.com/python/cpython/blob/3.12/Lib/pprint.py#L554-L633
    """
    typ = type(object)
    if typ in pprint._builtin_scalars:  # noqa: SLF001
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)

    if issubclass(typ, int) and r is int.__repr__:
        if self._underscore_numbers:
            return f"{object:_d}", True, False
        else:
            return repr(object), True, False

    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True  # noqa: SLF001
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        if self._sort_dicts:
            items = sorted(object.items(), key=pprint._safe_tuple)  # noqa: SLF001
        else:
            items = object.items()
        for k, v in items:
            krepr, kreadable, krecur = self.format(k, context, maxlevels, level)
            vrepr, vreadable, vrecur = self.format(v, context, maxlevels, level)
            append("%s: %s" % (krepr, vrepr))  # noqa: UP031
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or (
        issubclass(typ, tuple) and r is tuple.__repr__
    ):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True  # noqa: SLF001
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = self.format(o, context, maxlevels, level)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    if issubclass(typ, Recipe) and r is Recipe.__repr__:
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return f"{typ.__name__}(...)", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True  # noqa: SLF001
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object.steps:
            orepr, oreadable, orecur = self.format(o, context, maxlevels, level)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return f"{typ.__name__}({', '.join(components)})", readable, recursive

    if issubclass(typ, Step) and r is Step.__repr__:
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return f"{typ.__name__}(...)", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True  # noqa: SLF001
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        items = object._repr()  # noqa: SLF001
        for k, v in items:
            if k:
                krepr, kreadable, krecur = self.format(k, context, maxlevels, level)
                krepr = krepr.strip("'")
                middle = "="
            else:
                krepr, kreadable, krecur = "", True, False
                middle = ""
            vrepr, vreadable, vrecur = self.format(v, context, maxlevels, level)
            append(f"{krepr}{middle}{vrepr}")
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return f"{typ.__name__}({', '.join(components)})", readable, recursive

    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False
