import sys
from importlib import import_module, reload
from unittest.mock import patch

import pytest


# https://stackoverflow.com/a/65163627
@pytest.mark.parametrize("optional_dependency", ["numpy", "pandas", "pyarrow"])
def test_without_dependency(optional_dependency):
    with patch.dict(sys.modules, {optional_dependency: None}):
        if "ibis_ml" in sys.modules:
            reload(sys.modules["ibis_ml"])
        else:
            import_module("ibis_ml")
