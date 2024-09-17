import ibis
from ibis import _

import ibis_ml as ml


def test_expand_date():
    t = ibis.table({"x": "date", "y": "timestamp", "z": "int"})
    step = ml.ExpandDate(ml.date(), ("dow", "doy", "day", "week", "month", "year"))
    step.fit_table(t, ml.core.Metadata())

    res = step.transform_table(t)
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
    step.fit_table(t, ml.core.Metadata())

    res = step.transform_table(t)
    sol = t.mutate(
        x_hour=_.x.hour(),
        x_minute=_.x.minute(),
        x_second=_.x.second(),
        x_millisecond=_.x.millisecond(),
    )
    assert res.equals(sol)


def test_expand_timestamp():
    t = ibis.table({"y": "timestamp", "z": "int"})
    step = ml.ExpandTimestamp(
        ml.timestamp(),
        components=[
            "dow",
            "doy",
            "day",
            "week",
            "month",
            "year",
            "hour",
            "minute",
            "second",
            "millisecond",
        ],
    )
    step.fit_table(t, ml.core.Metadata())

    res = step.transform_table(t)
    sol = t.mutate(
        y_dow=_.y.day_of_week.index(),
        y_doy=_.y.day_of_year(),
        y_day=_.y.day(),
        y_week=_.y.week_of_year(),
        y_month=_.y.month() - 1,
        y_year=_.y.year(),
        y_hour=_.y.hour(),
        y_minute=_.y.minute(),
        y_second=_.y.second(),
        y_millisecond=_.y.millisecond(),
    )
    assert res.equals(sol)
