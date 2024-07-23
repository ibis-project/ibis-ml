"""IbisML is a library for building scalable ML pipelines using Ibis."""

__version__ = "0.1.2"

import pprint

from ibis_ml.core import Recipe, Step
from ibis_ml.select import (
    categorical,
    cols,
    contains,
    date,
    endswith,
    everything,
    floating,
    has_type,
    integer,
    matches,
    nominal,
    numeric,
    selector,
    startswith,
    string,
    temporal,
    time,
    timestamp,
    where,
)
from ibis_ml.steps import *
from ibis_ml.utils._pprint import _pprint_recipe, _pprint_step, _safe_repr
from ibis_ml.utils._split import train_test_split

# Add support for `Recipe`s and `Step`s to the built-in `PrettyPrinter`.
pprint.PrettyPrinter._dispatch[Recipe.__repr__] = _pprint_recipe  # noqa: SLF001
pprint.PrettyPrinter._dispatch[Step.__repr__] = _pprint_step  # noqa: SLF001
pprint.PrettyPrinter._safe_repr = _safe_repr  # noqa: SLF001


# Patch `skorch` since it does not support the array interface protocol.
def _auto_patch_skorch() -> None:
    try:
        import skorch.net
    except ImportError:
        return

    import ibis.expr.types as ir
    import numpy as np

    old_fit = skorch.net.NeuralNet.fit

    def fit(self, X, y=None, **fit_params):
        if isinstance(y, ir.Column):
            y = np.asarray(y)

        return old_fit(self, X, y, **fit_params)

    skorch.net.NeuralNet.fit = fit


_auto_patch_skorch()
