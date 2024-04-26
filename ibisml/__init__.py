from ibisml.core import Recipe, Step
from ibisml.select import (
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
from ibisml.steps import *

from ._version import __version__


def _auto_patch_skorch() -> None:
    import ibis.expr.types as ir
    import numpy as np
    import skorch.net

    old_fit = skorch.net.NeuralNet.fit

    def fit(self, X, y=None, **fit_params):
        if isinstance(y, ir.Column):
            y = np.asarray(y)

        return old_fit(self, X, y, **fit_params)

    skorch.net.NeuralNet.fit = fit


_auto_patch_skorch()
