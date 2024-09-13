import sys
from importlib import import_module
from unittest.mock import patch

import pytest


# https://stackoverflow.com/a/65034142/1093967
@pytest.mark.parametrize("optional_dependency", ["numpy", "pandas", "pyarrow"])
def test_without_dependency(optional_dependency):
    with patch.dict(sys.modules, {optional_dependency: None}):
        assert "ibis_ml" not in sys.modules
        import_module("ibis_ml")
