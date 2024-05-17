def _pprint_recipe(self, object, stream, indent, allowance, context, level):
    stream.write(object.__class__.__name__ + "(")
    if getattr(self, "_indent_at_name", True):
        indent += len(object.__class__.__name__)

    # TODO(deepyaman): Format parameters (beyond `steps`) more robustly.
    indent += self._indent_per_level
    k, v = "steps", list(object.steps)
    rep = self._repr(k, context, level)
    rep = rep.strip("'")
    middle = "="
    stream.write(rep)
    stream.write(middle)
    self._format(
        v, stream, indent + len(rep) + len(middle), allowance + 1, context, level
    )

    stream.write(")")
