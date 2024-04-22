import ibis
import pandas as pd
import pandas.testing as tm
import pytest

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
    expected = ibis.memtable(
        pd.DataFrame(
            {
                "time": pd.Series(
                    [
                        pd.Timestamp("2016-05-25 13:30:00.023"),
                        pd.Timestamp("2016-05-25 13:30:00.038"),
                        pd.Timestamp("2016-05-25 13:30:00.048"),
                        pd.Timestamp("2016-05-25 13:30:00.049"),
                        pd.Timestamp("2016-05-25 13:30:00.050"),
                        pd.Timestamp("2016-05-25 13:30:00.051"),
                    ]
                ),
                "ticker_AAPL": pd.Series([0, 0, 0, 0, 0, 0], dtype="Int8"),
                "ticker_GOOG": pd.Series([0, 0, 1, 1, 0, 0], dtype="Int8"),
                "ticker_MSFT": pd.Series([1, 1, 0, 0, 0, 0], dtype="Int8"),
                "ticker_None": pd.Series([0, 0, 0, 0, 0, 1], dtype="Int8"),
            }
        )
    )
    tm.assert_frame_equal(result.execute(), expected.execute())
