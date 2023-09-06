from textwrap import dedent

import ibis
import pyarrow as pa

import ibisml as ml


def test_categories_repr():
    c1 = ml.core.Categories(pa.array([]))
    assert repr(c1) == "Categories<ordered=False>"

    c2 = ml.core.Categories(pa.array(["a", "b"]))
    assert repr(c2) == "Categories<'a', 'b', ordered=False>"

    c3 = ml.core.Categories(pa.array(["a", "b", "c"]), ordered=True)
    assert repr(c3) == "Categories<'a', 'b', 'c', ordered=True>"

    c4 = ml.core.Categories(pa.array(["a", "b", "c", "d"]))
    assert repr(c4) == "Categories<'a', 'b', 'c', ..., ordered=False>"


def test_transform_result_repr():
    t = ibis.table({"a": "int", "b": "float", "c": "int", "d": "int"})

    res = ml.TransformResult(t, features=["a", "b"], outcomes=["c", "d"])
    assert repr(res) == dedent(
        """\
        TransformResult:
        - Features {
            a  int64
            b  float64
        }
        - Outcomes {
            c  int64
            d  int64
        }"""
    )

    res = ml.TransformResult(t, features=["a", "b", "c"])
    assert repr(res) == dedent(
        """\
        TransformResult:
        - Features {
            a  int64
            b  float64
            c  int64
        }"""
    )
