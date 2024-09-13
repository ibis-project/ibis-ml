import ibis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis_ml as ml


@pytest.fixture()
def train_table():
    return ibis.memtable(
        {
            "floating_col": [0.0, 0.0, 3.0, None, np.nan],
            "int_col": [0, 0, 3, None, None],
            "string_col": ["a", "a", "c", None, None],
            "null_col": [None] * 5,
        }
    )


@pytest.mark.parametrize(
    ("mode", "col_name", "expected"),
    [
        ("mean", "floating_col", 1.0),
        ("median", "floating_col", 0.0),
        ("mode", "floating_col", 0.0),
        ("mean", "int_col", 1),
        ("median", "int_col", 0),
        ("mode", "int_col", 0),
        ("mode", "string_col", "a"),
    ],
)
def test_impute(train_table, mode, col_name, expected):
    mode_class = getattr(ml, f"Impute{mode.capitalize()}")
    step = mode_class(col_name)
    test_table = ibis.memtable({col_name: [None]})
    step.fit_table(train_table, ml.core.Metadata())
    result = step.transform_table(test_table)
    expected = pd.DataFrame({col_name: [expected]})
    tm.assert_frame_equal(result.execute(), expected, check_dtype=False)


def test_fillna(train_table):
    step = ml.FillNA("floating_col", 0)
    step.fit_table(train_table, ml.core.Metadata())
    assert step.is_fitted()
    test_table = ibis.memtable({"floating_col": [None]})
    result = step.transform_table(test_table)
    expected = pd.DataFrame({"floating_col": [0]})
    tm.assert_frame_equal(result.execute(), expected, check_dtype=False)

    # test _fillna with None
    step = ml.FillNA("floating_col", None)
    step.fit_table(train_table, ml.core.Metadata())
    with pytest.raises(
        ValueError, match="Cannot fill column 'floating_col' with `None` or `NaN`"
    ):
        step.transform_table(test_table)
