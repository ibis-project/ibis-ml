import operator

import ibis
import pytest
from ibis import _

import ibisml as ml


@pytest.fixture()
def train_table():
    N = 100
    return ibis.memtable({"x": list(range(N)), "y": [10] * N, "z": ["s"] * N})


def test_PolynomialFeatures(train_table):
    step = ml.PolynomialFeatures(ml.numeric(), degree=2)
    step.fit_table(train_table, ml.core.Metadata())
    result_table = step.transform_table(train_table)
    sol = train_table.mutate(
        operator.pow(_.x, 2), operator.mul(_.x, _.y), operator.pow(_.y, 2)
    )
    assert sol.equals(result_table)
    # Check if the transformed table has the expected data
    for col_name in sol.columns:
        assert (
            sol[col_name].execute().tolist()
            == result_table[col_name].execute().tolist()
        )
