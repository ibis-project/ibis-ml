import ibis
import pytest

import ibisml as ml


@pytest.mark.parametrize(
    ("deviation_factor", "method", "treatment"),
    [
        (2, "z-score", "capping"),
        (2, "IQR", "capping"),
        (3.0, "z-score", "trimming"),
        (3.0, "IQR", "trimming"),
    ],
)
def test_UnivariateOutlier(deviation_factor, method, treatment):
    cols = {"col1": 0, "col2": 1}
    train_table = ibis.memtable(
        {
            # use same value for easier calculation statistics
            "col1": [cols["col1"]] * 10,  # mean = 0, std = 0
            "col2": [cols["col2"]] * 10,  # Q1 = 1, Q3 = 1
        }
    )

    test_table = ibis.memtable(
        {
            "col1": [
                None,  # keep
                cols["col1"],  # keep
                cols["col1"] - 1,  # outlier
                cols["col1"] + 1,  # outlier
                cols["col1"] + 1,  # outlier
            ],
            "col2": [
                cols["col2"],  # keep
                cols["col2"],  # keep
                cols["col2"] - 1,  # outlier
                cols["col2"] + 1,  # outlier
                None,  # keep
            ],
        }
    )
    step = ml.UnivariateOutlier(
        ml.numeric(),
        method=method,
        deviation_factor=deviation_factor,
        treatment=treatment,
    )
    step.fit_table(train_table, ml.core.Metadata())
    assert step.is_fitted()
    stats = step.stats_
    res = step.transform_table(test_table)

    if treatment == "trimming":
        assert res.count().execute() == 2
    elif treatment == "capping":
        assert res.count().execute() == 5

    for col_name, val in cols.items():
        # check the boundary
        assert stats[col_name]["lower_bound"] == val
        assert stats[col_name]["upper_bound"] == val
        # make sure there is no value beyond the boundary
        assert res[col_name].max().execute() <= val
        assert res[col_name].min().execute() >= val
