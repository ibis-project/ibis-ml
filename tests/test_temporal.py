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
        (
            ml.ExpandTime(ml.time()),
            "ExpandTime(time(), components=['hour', 'minute', 'second'])",
        ),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.ExpandDate(["x", "y"], ["dow", "month"]), "ExpandDate<x, y>"),
        (ml.transforms.ExpandTime(["x", "y"], ["hour"]), "ExpandTime<x, y>"),
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
        x_month=_.x.month() - 1,
        x_year=_.x.year(),
    )
    assert res.equals(sol)


def test_expand_time():
    t = ibis.table({"x": "time", "y": "timestamp", "z": "int"})
    step = ml.ExpandTime(ml.time(), ("hour", "minute", "second", "millisecond"))
    transform = step.fit(t, ml.core.Metadata())
    assert transform.columns == ["x"]

    res = transform.transform(t)
    sol = t.mutate(
        x_hour=_.x.hour(),
        x_minute=_.x.minute(),
        x_second=_.x.second(),
        x_millisecond=_.x.millisecond(),
    )
    assert res.equals(sol)
