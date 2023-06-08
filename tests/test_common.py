import pytest

import ibis
import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.Drop(ml.string()), "Drop(string())"),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.Drop(["x", "y"]), "Drop<x, y>"),
    ],
)
def test_transform_repr(transform, sol):
    assert repr(transform) == sol


def test_drop():
    t = ibis.table({"x": "int", "y": "float"})

    step = ml.Drop(ml.integer())
    transform = step.fit(t, ml.core.Metadata())

    assert isinstance(transform, ml.transforms.Drop)
    assert transform.columns == ["x"]

    res = transform.transform(t)
    assert res.equals(t.drop("x"))
