import ibis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis_ml as ml


@pytest.mark.parametrize(
    ("deviation_factor", "method", "treatment", "cols", "test_table", "expected"),
    [
        (
            2,
            "z-score",
            "capping",
            "int_col",
            {"int_col": [None, 0, -1, 1]},
            {"int_col": [None, 0, 0, 0]},
        ),
        (
            2,
            "IQR",
            "capping",
            "int_col",
            {"int_col": [None, 0, -1, 1]},
            {"int_col": [None, 0, 0, 0]},
        ),
        (
            3.0,
            "z-score",
            "trimming",
            "int_col",
            {"int_col": [None, 0, -1, 1]},
            {"int_col": [None, 0]},
        ),
        (
            3.0,
            "IQR",
            "trimming",
            "int_col",
            {"int_col": [None, 0, -1, 1]},
            {"int_col": [None, 0]},
        ),
        (
            2,
            "z-score",
            "capping",
            "floating_col",
            {"floating_col": [None, 0, -1, 1, np.nan]},
            {"floating_col": [None, 0.0, 0.0, 0.0, np.nan]},
        ),
        (
            2,
            "z-score",
            "trimming",
            "floating_col",
            {"floating_col": [None, 0, -1, 1, np.nan]},
            {"floating_col": [None, np.nan, 0.0]},
        ),
        (
            2,
            "z-score",
            "trimming",
            ["floating_col", "int_col"],
            {
                "floating_col": [None, 0, -1, 1, np.nan],
                "int_col": [None, 0, 0, None, None],
            },
            {"floating_col": [None, np.nan, 0.0], "int_col": [None, None, 0]},
        ),
        (
            2,
            "z-score",
            "capping",
            ["floating_col", "int_col"],
            {
                "floating_col": [None, 0, -1, 1, np.nan],
                "int_col": [None, 0, 0, None, None],
            },
            {
                "floating_col": [None, 0, 0, 0, np.nan],
                "int_col": [None, 0, 0, None, None],
            },
        ),
    ],
)
def test_handle_univariate_outliers(
    deviation_factor, method, treatment, cols, test_table, expected
):
    train_table = ibis.memtable(
        {
            # use same value for easier calculation statistics
            "int_col": [0] * 10,  # mean = 0, std = 0 Q1 = 0, Q3 = 0
            "floating_col": [0.0] * 10,  # mean = 0, std = 0 Q1 = 0, Q3 = 0
        }
    )

    test_table = ibis.memtable(test_table)
    step = ml.HandleUnivariateOutliers(
        cols, method=method, deviation_factor=deviation_factor, treatment=treatment
    )
    step.fit_table(train_table, ml.core.Metadata())
    assert step.is_fitted()

    result = step.transform_table(test_table)
    expected = pd.DataFrame(expected)

    tm.assert_frame_equal(result.execute(), expected, check_dtype=False)
