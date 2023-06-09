import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis import _

import ibisml as ml


def myfunc(col):
    pass


def myfunc2(col):
    pass


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.Drop(ml.string()), "Drop(string())"),
        (ml.Cast(ml.integer(), "float"), "Cast(integer(), 'float64')"),
        (ml.MutateAt(ml.integer(), myfunc), f"MutateAt(integer(), {myfunc!r})"),
        (ml.Mutate(myfunc, x=myfunc2), f"Mutate({myfunc!r}, x={myfunc2!r})"),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.Drop(["x", "y"]), "Drop<x, y>"),
        (ml.transforms.Cast(["x", "y"], dt.dtype("int")), "Cast<x, y>"),
        (ml.transforms.MutateAt(["x", "y"], myfunc), "MutateAt<x, y>"),
        (ml.transforms.Mutate(myfunc), "Mutate<...>"),
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


def test_mutate_at_expr():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.MutateAt(ml.integer(), _.abs())
    transform = step.fit(t, ml.core.Metadata())

    assert isinstance(transform, ml.transforms.MutateAt)
    assert transform.columns == ["x", "y"]

    res = transform.transform(t)
    sol = t.mutate(x=_.x.abs(), y=_.y.abs())
    assert res.equals(sol)


def test_mutate_at_named_exprs():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.MutateAt(ml.integer(), _.abs(), log=_.log())
    transform = step.fit(t, ml.core.Metadata())
    res = transform.transform(t)
    sol = t.mutate(x=_.x.abs(), y=_.y.abs(), x_log=_.x.log(), y_log=_.y.log())
    assert res.equals(sol)


def test_mutate():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.Mutate(_.x.abs().name("x_abs"), y_log=lambda t: t.y.log())
    transform = step.fit(t, ml.core.Metadata())
    res = transform.transform(t)
    sol = t.mutate(_.x.abs().name("x_abs"), y_log=lambda t: t.y.log())
    assert res.equals(sol)
