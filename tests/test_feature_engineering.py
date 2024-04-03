import ibis
from ibis import _

import ibisml as ml


def test_polynomial_features():
    t = ibis.table({"x": "int", "y": "float", "z": "string"})
    step = ml.PolynomialFeatures(ml.numeric(), degree=2)
    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    sol = t.mutate(
        poly_x_x=_.x * 1 * _.x,
        poly_x_y=_.x * 1 * _.y,
        poly_y_y=_.y * 1 * _.y
    )
    assert step.is_fitted()
    assert set(res.columns) == set(sol.columns)
    assert res.equals(sol)