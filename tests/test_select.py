import ibis
import ibis.expr.datatypes as dt
import pytest

import ibis_ml as ml


def eval_select(selector):
    metadata = ml.core.Metadata(targets=("y",))
    metadata.set_categories("a_categorical", ["a", "b"])
    metadata.set_categories("b_categorical", ["c", "d"])

    t = ibis.table(
        {
            "a_int": "int",
            "a_float": "float",
            "a_str": "str",
            "a_categorical": "int",
            "b_categorical": "int",
            "b_time": "time",
            "b_date": "date",
            "b_timestamp": "timestamp",
            "y": "int",
        }
    )

    return selector.select_columns(t, metadata)


def test_selector():
    s = ml.cols("x")
    assert ml.selector(s) is s

    assert ml.selector("x") == ml.cols("x")
    assert ml.selector(["x", "y"]) == ml.cols("x", "y")
    assert ml.selector(("x", "y")) == ml.cols("x", "y")

    def func(col):
        return col.get_name() == "foo"

    assert ml.selector(func) == ml.where(func)


def test_repr():
    assert repr(ml.integer()) == "integer()"
    assert repr(ml.matches("foo")) == "matches('foo')"


def test_everything():
    assert ml.everything() == ml.everything()
    assert eval_select(ml.everything()) == [
        "a_int",
        "a_float",
        "a_str",
        "a_categorical",
        "b_categorical",
        "b_time",
        "b_date",
        "b_timestamp",
    ]


def test_cols():
    assert ml.cols("x") == ml.cols("x")
    assert eval_select(ml.cols("a_int")) == ["a_int"]
    assert eval_select(ml.cols("a_int", "a_float")) == ["a_int", "a_float"]


def test_contains():
    assert ml.contains("a") == ml.contains("a")
    assert ml.contains("a") != ml.contains("b")

    assert eval_select(ml.contains("time")) == ["b_time", "b_timestamp"]


def test_endswith():
    assert ml.endswith("a") == ml.endswith("a")
    assert eval_select(ml.endswith("ical")) == ["a_categorical", "b_categorical"]


def test_startswith():
    assert ml.startswith("a") == ml.startswith("a")
    assert eval_select(ml.startswith("a_")) == [
        "a_int",
        "a_float",
        "a_str",
        "a_categorical",
    ]


def test_matches():
    assert ml.matches("a") == ml.matches("a")
    assert eval_select(ml.matches("ica")) == ["a_categorical", "b_categorical"]
    assert eval_select(ml.matches("$ica^")) == []


@pytest.mark.parametrize(
    ("selector", "cols"),
    [
        (ml.integer(), ["a_int"]),
        (ml.floating(), ["a_float"]),
        (ml.numeric(), ["a_int", "a_float"]),
        (ml.string(), ["a_str"]),
        (ml.time(), ["b_time"]),
        (ml.date(), ["b_date"]),
        (ml.timestamp(), ["b_timestamp"]),
        (ml.temporal(), ["b_time", "b_date", "b_timestamp"]),
        (ml.nominal(), ["a_str", "a_categorical", "b_categorical"]),
    ],
)
def test_type_selector(selector, cols):
    assert eval_select(selector) == cols


def test_categorical():
    assert ml.categorical() == ml.categorical()
    assert repr(ml.categorical()) == "categorical()"
    assert eval_select(ml.categorical()) == ["a_categorical", "b_categorical"]


def test_where():
    assert ml.where(bool) == ml.where(bool)
    assert ml.where(bool) != ml.where(lambda: False)
    assert eval_select(ml.where(lambda col: col.get_name() == "a_int")) == ["a_int"]


def test_has_type():
    assert ml.has_type("int") == ml.has_type("int")
    assert ml.has_type("int") == ml.has_type(dt.dtype("int"))

    assert eval_select(ml.has_type("int")) == ["a_int"]
    assert eval_select(ml.has_type(dt.int)) == ["a_int"]
    assert eval_select(ml.has_type(dt.Numeric)) == ["a_int", "a_float"]


def test_and():
    s1 = ml.startswith("a_") & ml.categorical()
    assert repr(s1) == "(startswith('a_') & categorical())"

    s2 = s1 & ml.endswith("foo")
    assert repr(s2) == "(startswith('a_') & categorical() & endswith('foo'))"

    assert eval_select(s1) == ["a_categorical"]
    assert eval_select(s2) == []


def test_or():
    s1 = ml.startswith("a_") | ml.contains("ica")
    assert repr(s1) == "(startswith('a_') | contains('ica'))"

    s2 = s1 | ml.endswith("stamp")
    assert repr(s2) == "(startswith('a_') | contains('ica') | endswith('stamp'))"

    assert eval_select(s1) == [
        "a_int",
        "a_float",
        "a_str",
        "a_categorical",
        "b_categorical",
    ]
    assert eval_select(s2) == [
        "a_int",
        "a_float",
        "a_str",
        "a_categorical",
        "b_categorical",
        "b_timestamp",
    ]


def test_and_or():
    s = (ml.integer() | ml.categorical()) & ml.startswith("a_")
    assert repr(s) == "((integer() | categorical()) & startswith('a_'))"
    assert eval_select(s) == ["a_int", "a_categorical"]


def test_and_or_implicit_cols():
    assert (ml.integer() & "x") == (ml.integer() & ml.cols("x"))
    assert (ml.integer() & ["x", "y"]) == (ml.integer() & ml.cols("x", "y"))

    assert (ml.integer() | "x") == (ml.integer() | ml.cols("x"))
    assert (ml.integer() | ["x", "y"]) == (ml.integer() | ml.cols("x", "y"))


def test_not():
    s1 = ml.numeric()
    s2 = ~s1
    assert ~s2 is s1
    assert repr(s2) == "~numeric()"
    assert eval_select(s2) == [
        "a_str",
        "a_categorical",
        "b_categorical",
        "b_time",
        "b_date",
        "b_timestamp",
    ]
