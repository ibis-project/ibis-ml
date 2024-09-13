import sys
from importlib import import_module, reload
from unittest import mock

import pytest


@pytest.mark.parametrize("optional_dep", ["pandas", "numpy", "pyarrow"])
def test_optional_dependencies(optional_dep):
    with mock.patch.dict(sys.modules, {"optional_dependency": None}):
        if "ibis_ml" in sys.modules:
            reload(sys.modules["ibis_ml"])
        else:
            import_module("ibis_ml")

        assert "ibis_ml" in sys.modules
