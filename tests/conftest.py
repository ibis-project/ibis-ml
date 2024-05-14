import pytest
from ibis.backends import _get_backend_names
from ibis.backends.conftest import pytest_runtest_call  # noqa: F401

ALL_BACKENDS = set(_get_backend_names())


@pytest.fixture(params=ALL_BACKENDS, scope="session")
def backend(request):
    return request.param
