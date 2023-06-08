import pytest

import ibis
import ibis.expr.datatypes as dt

import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.Drop(ml.string()), "Drop(string())"),
        (ml.Cast(ml.integer(), "float"), "Cast(integer(), Float64(nullable=True))"),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.Drop(["x", "y"]), "Drop<x, y>"),
        (ml.transforms.Cast(["x", "y"], dt.dtype("int")), "Cast<x, y>"),
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


def test_cast():
    t = ibis.table({"x": "int", "y": "float"})

    step = ml.Cast(ml.integer(), "float")
    assert step.dtype == dt.float64

    transform = step.fit(t, ml.core.Metadata())

    assert isinstance(transform, ml.transforms.Cast)
    assert transform.columns == ["x"]

    res = transform.transform(t)
    sol = t.mutate(x=t.x.cast("float"))
    assert res.equals(sol)
