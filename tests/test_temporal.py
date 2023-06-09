import pytest

import ibis
from ibis import _

import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (
            ml.ExpandDate(ml.date()),
            "ExpandDate(date(), components=['dow', 'month', 'year'])",
        ),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.ExpandDate(["x", "y"], ["dow", "month"]), "ExpandDate<x, y>"),
    ],
)
def test_transform_repr(transform, sol):
    assert repr(transform) == sol


def test_expand_date():
    t = ibis.table({"x": "date", "y": "timestamp", "z": "int"})
    step = ml.ExpandDate(ml.date(), ("dow", "doy", "day", "week", "month", "year"))
    transform = step.fit(t, ml.core.Metadata())
    assert transform.columns == ["x"]

    res = transform.transform(t)
    sol = t.mutate(
        x_dow=_.x.day_of_week.index(),
        x_doy=_.x.day_of_year(),
        x_day=_.x.day(),
        x_week=_.x.week_of_year(),
        x_month=_.x.month(),
        x_year=_.x.year(),
    )
    assert res.equals(sol)
