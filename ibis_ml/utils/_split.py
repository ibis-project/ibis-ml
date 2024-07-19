import random

import ibis
import ibis.expr.types as ir
from ibis import _


def train_test_split(
    table: ir.Table,
    unique_key: str | list[str],
    test_size: float = 0.25,
    num_buckets: int = 100,
    random_seed: int | None = None,
) -> tuple[ir.Table, ir.Table]:
    """Randomly split Ibis table data into training and testing tables.

    This function splits an Ibis table into training and testing tables
    based on a unique key or combination of keys. It uses a hashing function to
    convert the unique key into an integer, then applies a modulo operation to split
    the data into buckets. The training table consists of data points from a subset of
    these buckets, while the remaining data points form the test table.

    Parameters
    ----------
    table
        The input Ibis table to be split.
    unique_key
        The column name(s) that uniquely identify each row in the table. This unique_key
        is used to create a deterministic split of the dataset through a hashing
        process.
    test_size
        The ratio of the dataset to include in the test split, which should be between
        0 and 1. This ratio is approximate because the hashing algorithm may not provide
        a uniform bucket distribution for small datasets. Larger datasets will result in
        more uniform bucket assignments, making the split ratio closer to the desired
        value.
    num_buckets
        The number of buckets into which the data is divided during the splitting
        process. It controls how finely the data is divided into buckets during
        the split process. Adjusting num_buckets can affect the granularity and
        efficiency of the splitting operation, balancing between accuracy and
        computational efficiency.
    random_seed
        Seed for the random number generator. If provided, ensures reproducibility
        of the split.

    Returns
    -------
    tuple[ir.Table, ir.Table]
        A tuple containing two Ibis tables: (train_table, test_table).

    Raises
    ------
    ValueError
        If test_size is not a float between 0 and 1.

    Examples
    --------
    >>> import ibis_ml as ml

    Split an Ibis table into training and testing tables.

    >>> table = ibis.memtable({"key1": range(100)})
    >>> train_table, test_table = ml.train_test_split(
    ...     table,
    ...     unique_key="key1",
    ...     test_size=0.2,
    ...     random_seed=0,
    ... )
    """
    if not (0 < test_size < 1):
        raise ValueError("test size should be a float between 0 and 1.")

    # Set the random seed for reproducibility
    if random_seed:
        random.seed(random_seed)

    # Generate a random 256-bit key
    random_key = str(random.getrandbits(256))

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    table = table.mutate(
        combined_key=ibis.literal(",").join(
            table[col].cast("str") for col in unique_key
        )
    ).mutate(
        train=(_.combined_key + random_key).hash().abs() % num_buckets
        < int((1 - test_size) * num_buckets)
    )

    return table[table.train].drop(["combined_key"]), table[~table.train].drop(
        ["combined_key"]
    )
