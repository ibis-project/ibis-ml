import ibis
import pandas.testing as tm

import ibis_ml as ml


def test_train_test_split():
    N = 100
    test_size = 0.25
    table = ibis.memtable({"key1": range(N)})

    train_table, test_table = ml.train_test_split(
        table, unique_key="key1", test_size=test_size, random_seed=42
    )

    # Check counts and overlaps in train and test dataset
    assert train_table.count().execute() + test_table.count().execute() == N
    assert train_table.intersect(test_table).count().execute() == 0

    # Check reproducibility
    reproduced_train_table, reproduced_test_table = ml.train_test_split(
        table, unique_key="key1", test_size=test_size, random_seed=42
    )
    tm.assert_frame_equal(train_table.execute(), reproduced_train_table.execute())
    tm.assert_frame_equal(test_table.execute(), reproduced_test_table.execute())

    # make sure it could generate different data with different random_seed
    different_train_table, different_test_table = ml.train_test_split(
        table, unique_key="key1", test_size=test_size, random_seed=0
    )
    assert not train_table.execute().equals(different_train_table.execute())
    assert not test_table.execute().equals(different_test_table.execute())
