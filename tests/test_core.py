import ibisml as ml

import pandas as pd
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

    assert isinstance(res, pd.DataFrame)
    sol = df.assign(d=df.a + df.b).drop("c", axis=1)
    assert res.equals(sol)
