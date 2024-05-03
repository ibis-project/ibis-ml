import ibis
import ibis.expr.datatypes as dt
from ibis import _

import ibis_ml as ml


def test_drop():
    t = ibis.table({"x": "int", "y": "float"})

    step = ml.Drop(ml.integer())
    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    assert res.equals(t.drop("x"))


def test_cast():
    t = ibis.table({"x": "int", "y": "float"})

    step = ml.Cast(ml.integer(), "float")
    assert step.dtype == dt.float64

    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    sol = t.mutate(x=t.x.cast("float"))
    assert res.equals(sol)


def test_mutate_at_expr():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.MutateAt(ml.integer(), _.abs())
    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    sol = t.mutate(x=_.x.abs(), y=_.y.abs())
    assert res.equals(sol)


def test_mutate_at_named_exprs():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.MutateAt(ml.integer(), _.abs(), log=_.log())
    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    sol = t.mutate(x=_.x.abs(), y=_.y.abs(), x_log=_.x.log(), y_log=_.y.log())
    assert res.equals(sol)


def test_mutate():
    t = ibis.table({"x": "int", "y": "int", "z": "string"})

    step = ml.Mutate(_.x.abs().name("x_abs"), y_log=lambda t: t.y.log())
    step.fit_table(t, ml.core.Metadata())
    res = step.transform_table(t)
    sol = t.mutate(_.x.abs().name("x_abs"), y_log=lambda t: t.y.log())
    assert res.equals(sol)
