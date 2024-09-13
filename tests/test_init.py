import sys
from unittest.mock import patch

import pytest


# https://stackoverflow.com/a/65034142/1093967
@pytest.mark.parametrize("optional_dependency", ["pandas", "numpy", "pyarrow"])
def test_without_dependency(optional_dependency):
    with patch.dict(sys.modules, {optional_dependency: None}):
        assert "ibis_ml" not in sys.modules
        import ibis_ml  # noqa: F401
