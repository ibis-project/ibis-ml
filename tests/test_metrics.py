import ibis
import pytest
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score

from ibis_ml.metrics import accuracy_score, f1_score, precision_score, recall_score


@pytest.fixture
def results_table():
    return ibis.memtable(
        {
            "id": range(1, 13),
            "actual": [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "prediction": [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_accuracy_score(results_table):
    t = results_table
    df = t.to_pandas()
    result = accuracy_score(t.actual, t.prediction)
    expected = sk_accuracy_score(df["actual"], df["prediction"])
    assert result == pytest.approx(expected, abs=1e-4)


def test_precision_score(results_table):
    t = results_table
    df = t.to_pandas()
    result = precision_score(t.actual, t.prediction)
    expected = sk_precision_score(df["actual"], df["prediction"])
    assert result == pytest.approx(expected, abs=1e-4)


def test_recall_score(results_table):
    t = results_table
    df = t.to_pandas()
    result = recall_score(t.actual, t.prediction)
    expected = sk_recall_score(df["actual"], df["prediction"])
    assert result == pytest.approx(expected, abs=1e-4)


def test_f1_score(results_table):
    t = results_table
    df = t.to_pandas()
    result = f1_score(t.actual, t.prediction)
    expected = sk_f1_score(df["actual"], df["prediction"])
    assert result == pytest.approx(expected, abs=1e-4)
