import ibis
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from ibis import _

import ibisml as ml


class Shuffle(ml.Step):
    """A Step to reorder the table, for use in testing that order is maintained"""

    def is_fitted(self):
        return True

    def transform_table(self, table):
        return table.order_by(ibis.random())


@pytest.fixture()
def df():
    return pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 0, 1, 0, 1], "c": ["x", "x", "y", "x", "y"]}
    )


@pytest.fixture()
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
    r = ml.Recipe(ml.Mutate(d=_.a + _.b), ml.Drop(~ml.numeric()))

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

    with pytest.raises(
        ValueError, match=r"`transform` must be one of \(.*\), got 'unsupported'"
    ):
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

    out2 = r.fit_transform(table)
    assert isinstance(out2, typ)


def test_to_numpy_errors_non_numeric(table):
    r = ml.Recipe(ml.Mutate(d=_.a + _.b))
    r.fit(table)
    with pytest.raises(ValueError, match="Not all output columns are numeric"):
        r.to_numpy(table)


@pytest.mark.parametrize(
    "format", ["numpy", "pandas", "pyarrow", "polars", "ibis-table"]
)
def test_input_formats(format):
    r = ml.Recipe(ml.Cast(ml.everything(), "float64"))
    X = np.eye(3, dtype="i8")
    if format == "polars":
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(X, schema=["x0", "x1", "x2"])
    elif format != "numpy":
        X = pd.DataFrame(X, columns=["x0", "x1", "x2"])
        if format == "ibis-table":
            X = ibis.memtable(X)
        elif format == "pyarrow":
            X = pa.Table.from_pandas(X)
    r.fit(X)
    out = r.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == "f8"


@pytest.mark.parametrize(
    "format", ["numpy", "pandas", "pyarrow", "polars", "ibis-table"]
)
def test_transform_in_memory_data_maintains_order(format):
    r = ml.Recipe(ml.Cast(ml.everything(), "float64"), Shuffle())

    X = np.vstack([np.arange(100), np.arange(100, 200)]).T
    if format == "polars":
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(X, schema=["x0", "x1"])
    elif format != "numpy":
        X = pd.DataFrame(X, columns=["x0", "x1"])
        if format == "ibis-table":
            X = ibis.memtable(X)
        elif format == "pyarrow":
            X = pa.Table.from_pandas(X)
    r.fit(X)
    out = r.to_pandas(X)

    # table inputs won't maintain order, in-memory inputs will
    should_be_ordered = format != "ibis-table"

    assert list(out.columns) == ["x0", "x1"]

    for col in out.columns:
        assert out[col].dtype == "f8"
        assert out[col].is_monotonic_increasing == should_be_ordered


def test_can_use_in_sklearn_pipeline():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    X = np.array([[1, 3], [2, 4], [3, 5]])
    y = np.array([10, 11, 12])

    r = ml.Recipe(ml.Mutate(x2=_.x0 + _.x1), ml.ScaleStandard(ml.everything()))
    p = Pipeline([("recipe", r), ("model", LinearRegression())])

    # get/set params works
    params = p.get_params()
    p.set_params(**params)

    # fit and predict work
    p.fit(X, y)
    assert isinstance(p.predict(X), np.ndarray)

    # clone works
    p2 = sklearn.clone(p)
    r2 = p2.named_steps["recipe"]
    assert r2 is not r
    assert not r2.is_fitted()
