import ibis
import ibis.common.exceptions as com
import pytest

import ibis_ml as ml


@pytest.mark.notimpl(
    [
        "bigquery",
        "mysql",
        "sqlite",
        "mssql",
        "trino",
        "flink",
        "druid",
        "datafusion",
        "impala",
        "exasol",
    ],
    raises=com.OperationNotDefinedError,
    reason="Quantile is not supported",
)
@pytest.mark.notimpl(
    ["pandas", "polars", "dask"],
    raises=NotImplementedError,
    reason="Backend doesn't support SQL",
)
def test_discretize_quantile(backend):
    train_table = ibis.memtable({"col": range(1, 11)})
    step = ml.DiscretizeKBins("col", n_bins=9, strategy="quantile")
    step.fit_table(train_table, ml.core.Metadata())
    t = step.transform_table(train_table)

    fit_table_sql = ibis.to_sql(step.expr.op().to_expr(), dialect=backend)
    assert fit_table_sql is not None

    transform_table_sql = ibis.to_sql(t)
    assert transform_table_sql is not None


@pytest.mark.notimpl(
    ["pandas", "polars", "dask"],
    raises=NotImplementedError,
    reason="Backend doesn't support SQL",
)
def test_discretize_uniform(backend):
    train_table = ibis.memtable({"col": range(1, 11)})
    step = ml.DiscretizeKBins("col", n_bins=9, strategy="uniform")
    step.fit_table(train_table, ml.core.Metadata())
    t = step.transform_table(train_table)

    fit_table_sql = ibis.to_sql(step.expr.op().to_expr(), dialect=backend)
    assert fit_table_sql is not None

    transform_table_sql = ibis.to_sql(t)
    assert transform_table_sql is not None


@pytest.mark.notimpl(
    ["pandas", "polars", "dask"],
    raises=NotImplementedError,
    reason="Backend doesn't support SQL",
)
def test_create_polynomial_features(backend):
    N = 10
    train_table = ibis.memtable({"x": list(range(N)), "y": [10] * N})
    step = ml.CreatePolynomialFeatures(ml.numeric(), degree=2)
    step.fit_table(train_table, ml.core.Metadata())
    t = step.transform_table(train_table)
    transform_table_sql = ibis.to_sql(t)
    assert transform_table_sql is not None
