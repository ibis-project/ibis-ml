import ibis
import ibis.selectors as s
import pytest

import ibisml as ml


@pytest.fixture()
def train_table():
    N = 100
    return ibis.memtable(
        {
            "variable_col": list(range(N)),
            "constant_col": [1.0] * N,
            "str_col": ["value"] * N,
        }
    )


@pytest.fixture()
def test_table():
    return ibis.memtable(
        {
            "variable_col": [float("-inf"), 0, 1, 12, float("inf")],
            "constant_col": [float("-inf"), 1.0, 1.0, 1.0, float("inf")],
            "str_col": ["value"] * 5,
        }
    )


@pytest.mark.parametrize("k", [2, 5, 10, 100, 1000])
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_KBinsDiscretizer(train_table, test_table, k, strategy):
    step = ml.KBinsDiscretizer(
        ml.numeric(), n_bins=k, strategy=strategy, overwrite=True
    )
    step.fit_table(train_table, ml.core.Metadata())
    train_res = step.transform_table(train_table)
    test_res = step.transform_table(test_table)
    assert step.is_fitted()
    assert len(step.bins_edge_) == len(train_table.select(s.of_type("numeric")).columns)
    for edges in step.bins_edge_.values():
        assert all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1))
        assert len(edges) == k + 1
    assert test_res.columns == test_table.columns
    assert test_res.count().execute() == test_table.count().execute()
    for col_name in test_res.select(s.of_type("numeric")).columns:
        assert test_res[col_name].max().execute() < k
        assert test_res[col_name].min().execute() >= 0
        assert 1 <= test_res[col_name].nunique().execute() <= k
    assert train_res["constant_col"].nunique().execute() == 1
    assert test_res["constant_col"].nunique().execute() > 1


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_KBinsDiscretizer_overwrite(train_table, test_table, overwrite, strategy):
    ibis.options.interactive = True
    k = 10
    step = ml.KBinsDiscretizer(
        ml.numeric(), n_bins=k, strategy=strategy, overwrite=overwrite
    )
    step.fit_table(train_table, ml.core.Metadata())
    test_res = step.transform_table(test_table)
    assert len(test_res.select(s.of_type("numeric")).columns) == len(
        test_table.select(s.of_type("numeric")).columns
    ) * (1 if overwrite else 2)
