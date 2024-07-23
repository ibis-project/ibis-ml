import ibis
import numpy as np
import pandas as pd

import ibis_ml as ml


def test_drop_zero_variance():
    zv_numeric_col = [1.0] * 10
    all_null_numeric_col = [None] * 10
    all_nan_numeric_col = [np.nan] * 10
    non_zv_numeric_col = list(range(10))
    zv_string_col = ["String"] * 10
    non_zv_string_col = [f"String_{i}" for i in range(10)]
    start_timestamp = pd.Timestamp("2000-01-01 00:00:00.000")
    zv_timestamp_col = [start_timestamp] * 10
    non_zv_timestamp_col = [
        start_timestamp + pd.Timedelta(minutes=i) for i in range(10)
    ]

    zv_cols = {
        "zero_variance_numeric_col",
        "zero_variance_string_col",
        "zero_variance_timestamp_col",
        "all_null_numeric_col",
        "all_nan_numeric_col",
    }

    t_train = ibis.memtable(
        {
            "zero_variance_numeric_col": zv_numeric_col,
            "all_null_numeric_col": all_null_numeric_col,
            "all_nan_numeric_col": all_nan_numeric_col,
            "non_zero_variance_numeric_col": non_zv_numeric_col,
            "zero_variance_string_col": zv_string_col,
            "non_zero_variance_string_col": non_zv_string_col,
            "zero_variance_timestamp_col": zv_timestamp_col,
            "non_zero_variance_timestamp_col": non_zv_timestamp_col,
        }
    )
    t_test = ibis.memtable(
        {
            "zero_variance_numeric_col": [],
            "all_null_numeric_col": [],
            "all_nan_numeric_col": [],
            "non_zero_variance_numeric_col": [],
            "zero_variance_string_col": [],
            "non_zero_variance_string_col": [],
            "zero_variance_timestamp_col": [],
            "non_zero_variance_timestamp_col": [],
        }
    )

    step = ml.DropZeroVariance(ml.everything())
    step.fit_table(t_train, ml.core.Metadata())
    res = step.transform_table(t_test)
    sol = t_test.drop(zv_cols)
    # TODO(deepyaman): Revert to `assert sol.equals(res)` pending
    #   https://github.com/ibis-project/ibis/pull/9677
    assert sol.schema() == res.schema()
