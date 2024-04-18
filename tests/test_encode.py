from functools import reduce

import ibis
import pandas as pd
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
            # AMZN is unkown category for training dataset
            "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AMZN", None],
        }
    )


def test_count_encode(t_train, t_test):
    step = ml.CountEncode("ticker")
    step.fit_table(t_train, ml.core.Metadata())
    res = step.transform_table(t_test)
    assert res.to_pandas().sort_values(by="time").ticker.to_list() == [4, 4, 2, 2, 0, 0]


def test_onehotencode(t_train, t_test):
    col = "ticker"
    step = ml.OneHotEncode(col)
    step.fit_table(t_train, ml.core.Metadata())
    result = step.transform_table(t_test)

    encoded_cols = [
        f"{col}_{v!s}"
        for v in t_train.select("ticker").distinct().ticker.to_pyarrow().to_pylist()
    ]

    # Check the number of columns
    assert (
        len(result.columns) == len(t_test.columns) + len(encoded_cols) - 1
    ), "Incorrect number of encoded columns"

    # Ensure all encoded columns are present
    assert set(result.columns).issuperset(
        set(encoded_cols)
    ), "Not all encoded columns are present"

    # Verify that each encoded value is either 0 or 1
    assert all(
        ((result[col_name] == 0) | (result[col_name] == 1)).all().execute()
        for col_name in encoded_cols
    ), "Encoded values are not all 0 or 1"

    # Check the sum of all encoded columns for each row
    result = result.mutate(
        sum_encode_per_row=reduce(
            lambda acc, col_name: acc + result[col_name], encoded_cols, 0
        )
    )
    assert result.to_pandas().sum_encode_per_row.to_list() == [
        1,
        1,
        1,
        1,
        0,  # The 5th row's ticker "AMZN" is unknown for the training data
        1,
    ], "Incorrect sum of encoded columns per row per feature"
