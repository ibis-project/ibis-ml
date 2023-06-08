from textwrap import dedent

import ibis
import ibisml as ml


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
