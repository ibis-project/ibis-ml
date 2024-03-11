import ibisml as ml

import numpy as np
import pandas as pd
import pyarrow as pa
import ibis
from ibis import _
import pytest


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 0, 1, 0, 1],
            "c": ["x", "x", "y", "x", "y"],
        }
    )


@pytest.fixture
def table(df):
    return ibis.memtable(df)


def test_is_fitted(table):
    r = ml.Recipe(ml.Drop(~ml.numeric()))
    assert not r.is_fitted()

    r.fit(table)
    assert r.is_fitted()


def test_sklearn_clone(table):
    sklearn = pytest.importorskip("sklearn")

    r1 = ml.Recipe(ml.Drop(~ml.numeric()))
    assert not r1.is_fitted()

    r1.fit(table)
    assert r1.is_fitted()

    r2 = sklearn.clone(r1)
    assert not r2.is_fitted()

    r2.fit(table)
    assert r1.to_table(table).equals(r2.to_table(table))


def test_in_memory_workflow(df):
    r = ml.Recipe(
        ml.Mutate(d=_.a + _.b),
        ml.Drop(~ml.numeric()),
    )

    r.fit(df)
    assert r.is_fitted()
    res = r.transform(df)

    assert isinstance(res, np.ndarray)
    sol = df.assign(d=df.a + df.b).drop("c", axis=1).values
    np.testing.assert_array_equal(res, sol)


def test_set_output():
    recipe = ml.Recipe(ml.Drop(~ml.numeric()))
    assert recipe.output_format == "default"

    for format in ["default", "pandas", "pyarrow", "polars"]:
        assert recipe.set_output(transform=format) is recipe
        assert recipe.output_format == format

    recipe.set_output(transform="polars")  # something non-standard
    recipe.set_output(transform=None)  # None -> leave unchanged
    assert recipe.output_format == "polars"

    with pytest.raises(ValueError):
        recipe.set_output(transform="unsupported")


@pytest.mark.parametrize("format", ["pandas", "polars", "pyarrow", "default"])
def test_output_formats(table, format):
    if format == "pandas":
        typ = pd.DataFrame
    elif format == "pyarrow":
        typ = pa.Table
    elif format == "polars":
        typ = pytest.importorskip("polars").DataFrame
    else:
        typ = np.ndarray

    r = ml.Recipe(ml.Mutate(d=_.a + _.b), ml.Drop(~ml.numeric()))
    r.set_output(transform=format)
    r.fit(table)
    out = r.transform(table)
    assert isinstance(out, typ)


def test_to_numpy_errors_non_numeric(table):
    r = ml.Recipe(ml.Mutate(d=_.a + _.b))
    r.fit(table)
    with pytest.raises(ValueError, match="Not all output columns are numeric"):
        r.to_numpy(table)


@pytest.mark.parametrize("format", ["numpy", "pandas", "pyarrow", "polars", "ibis-table"])
def test_input_formats(format):
    r = ml.Recipe(ml.Cast(ml.everything(), "float64"))
    X = np.eye(3, dtype="i8")
    if format == "polars":
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(np.eye(3, dtype="i8"), schema=["x1", "x2", "x3"])
    elif format != "numpy":
        X = pd.DataFrame(np.eye(3, dtype="i8"), columns=["x1", "x2", "x3"])
        if format == "ibis-table":
            X = ibis.memtable(X, name="test")
        elif format == "pyarrow":
            X = pa.Table.from_pandas(X)
    r.fit(X)
    out = r.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == "f8"
