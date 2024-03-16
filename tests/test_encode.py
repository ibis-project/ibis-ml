import ibis
import pandas as pd

import ibisml as ml


def test_count_encode():
    t_train = ibis.memtable(
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
    t_test = ibis.memtable(
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

    step = ml.CountEncode("ticker")
    step.fit_table(t_train, ml.core.Metadata())
    res = step.transform_table(t_test)
    assert res.to_pandas().sort_values(by="time").ticker.to_list() == [4, 4, 2, 2, 0, 0]
