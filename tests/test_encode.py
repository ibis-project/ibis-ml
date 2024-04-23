import ibis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from sklearn.preprocessing import TargetEncoder

import ibisml as ml


@pytest.fixture()
def t_train():
    return ibis.memtable(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", None, "AAPL", "GOOG", "MSFT"],
        }
    )


@pytest.fixture()
def t_test():
    return ibis.memtable(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.050"),
                pd.Timestamp("2016-05-25 13:30:00.051"),
            ],
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AMZN", None],
        }
    )


def test_count_encode(t_train, t_test):
    step = ml.CountEncode("ticker")
    step.fit_table(t_train, ml.core.Metadata())
    res = step.transform_table(t_test)
    assert res.to_pandas().sort_values(by="time").ticker.to_list() == [4, 4, 2, 2, 0, 0]


def test_one_hot_encode(t_train, t_test):
    step = ml.OneHotEncode("ticker")
    step.fit_table(t_train, ml.core.Metadata())
    result = step.transform_table(t_test)
    expected = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.050"),
                pd.Timestamp("2016-05-25 13:30:00.051"),
            ],
            "ticker_AAPL": [0, 0, 0, 0, 0, 0],
            "ticker_GOOG": [0, 0, 1, 1, 0, 0],
            "ticker_MSFT": [1, 1, 0, 0, 0, 0],
            "ticker_None": [0, 0, 0, 0, 0, 1],
        }
    )
    tm.assert_frame_equal(result.execute(), expected, check_dtype=False)


@pytest.mark.parametrize("smooth", [5000.0, 1.0, 0.0])
def test_target_encode(smooth):
    data = pd.DataFrame(
        {
            "X": ["dog"] * 20 + ["cat"] * 30 + ["snake"] * 38,
            "y": [90.3] * 5
            + [80.1] * 15
            + [20.4] * 5
            + [20.1] * 25
            + [21.2] * 8
            + [49] * 30,
        }
    )
    X = data[["X"]]
    y = data.y
    t_train = ibis.memtable(data)

    enc = TargetEncoder(smooth=smooth).fit(X, y)
    expected = pd.DataFrame(
        {
            "X": np.append(enc.categories_[0], "ibis"),
            "expected": np.append(enc.encodings_[0], y.mean()),
        }
    )
    t_test = ibis.memtable(expected)

    step = ml.TargetEncode("X", smooth)
    step.fit_table(t_train, ml.core.Metadata(targets=("y",)))
    res = step.transform_table(t_test).to_pandas()
    assert np.allclose(res.X, res.expected)
