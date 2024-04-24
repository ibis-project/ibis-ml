import ibis
import pandas as pd
import pandas.testing as tm
import pytest

import ibisml as ml


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_discretize(strategy):
    col = "col"
    k = 9
    train_table = ibis.memtable({col: range(1, 11)})
    variable_col_data = [float("-inf"), 1.5, 2.5, 3.5, 8.5, float("inf")]
    test_table = ibis.memtable({col: variable_col_data})
    expected = pd.DataFrame(
        {col: variable_col_data, f"{col}_{k}_bin_{strategy}": [0, 0, 1, 2, 7, 8]}
    )

    step = ml.DiscretizeKBins(col, n_bins=k, strategy=strategy)
    step.fit_table(train_table, ml.core.Metadata())
    result = step.transform_table(test_table)

    tm.assert_frame_equal(result.execute(), expected, check_dtype=False)
