import ibis
import pandas.testing as tm
import pytest

import ibisml as ml


@pytest.fixture()
def train_table():
    N = 100
    return ibis.memtable({"x": list(range(N)), "y": [10] * N, "z": ["s"] * N})


def test_PolynomialFeatures(train_table):
    step = ml.PolynomialFeatures(ml.numeric(), degree=2)
    step.fit_table(train_table, ml.core.Metadata())
    result = step.transform_table(train_table)
    expected = train_table.mutate(
        **{
            "poly_x^2": train_table.x**2,
            "poly_x_y": train_table.x * train_table.y,
            "poly_y^2": train_table.y**2,
        }
    )
    # Check if the transformed table has the expected data
    tm.assert_frame_equal(result.execute(), expected.execute())
