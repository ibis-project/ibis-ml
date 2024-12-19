import ibis
import pytest
import sklearn.metrics

import ibis_ml.metrics


@pytest.fixture
def results_table():
    return ibis.memtable(
        {
            "id": range(1, 13),
            "actual": [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "prediction": [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize(
    "metric_name",
    [
        pytest.param("accuracy_score", id="accuracy_score"),
        pytest.param("precision_score", id="precision_score"),
        pytest.param("recall_score", id="recall_score"),
        pytest.param("f1_score", id="f1_score"),
    ],
)
def test_classification_metrics(results_table, metric_name):
    ibis_ml_func = getattr(ibis_ml.metrics, metric_name)
    sklearn_func = getattr(sklearn.metrics, metric_name)
    t = results_table
    df = t.to_pandas()
    result = ibis_ml_func(t.actual, t.prediction).to_pyarrow().as_py()
    expected = sklearn_func(df["actual"], df["prediction"])
    assert result == pytest.approx(expected, abs=1e-4)
