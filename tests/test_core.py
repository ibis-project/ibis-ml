from textwrap import dedent

import ibis
import pandas as pd
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

    res = ml.TransformResult(t, features=["a", "b", "c"], other=["d"])
    assert repr(res) == dedent(
        """\
        TransformResult:
        - Features {
            a  int64
            b  float64
            c  int64
        }
        - Other {
            d  int64
        }"""
    )


def test_categorize_pandas():
    categories = {
        "num_col": ml.core.Categories(pa.array([1, 2, 3, 4, 5]), ordered=True),
        "str_col": ml.core.Categories(pa.array(["a", "b", "c"]), ordered=False),
    }

    df = pd.DataFrame(
        {
            "num_col": [
                0,
                1,
                2,
                -1,
                4,
            ],
            "str_col": [
                0,
                1,
                2,
                0,
                -1,
            ],
        }
    )

    t = ibis.table({"num_col": "int64", "str_col": "string"})
    res = ml.TransformResult(t, categories=categories)

    transformed_df = res._categorize_pandas(df)

    assert isinstance(transformed_df["num_col"].dtype, pd.CategoricalDtype)
    assert isinstance(transformed_df["str_col"].dtype, pd.CategoricalDtype)
    assert transformed_df["num_col"].cat.ordered is True
    assert transformed_df["str_col"].cat.ordered is False
    assert transformed_df["num_col"].isna().sum() == 1
    assert transformed_df["str_col"].isna().sum() == 1
