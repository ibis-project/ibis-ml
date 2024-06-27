import random

import ibis
import ibis.expr.types as ir
from ibis import _


def train_test_split(
    table: ir.Table,
    unique_key: str | list[str],
    test_size: float = 0.25,
    random_state=42,
) -> tuple[ir.Table, ir.Table]:

    if not (0 < test_size < 1):
        raise ValueError("test size should be a float between 0 and 1.")

    # Set the random seed for reproducibility
    random.seed(random_state)
    # Generate a random 256-bit key
    random_key = str(random.getrandbits(256))
    # set the number of buckets
    num_buckets = 100000

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    table = table.mutate(
        combined_key=ibis.literal("").join(table[col].cast("str") for col in unique_key)
    ).mutate(
        train=(_.combined_key + random_key).hash().abs() % num_buckets
        < int((1 - test_size) * num_buckets)
    )

    return table[table.train].drop(["combined_key"]), table[~table.train].drop(
        ["combined_key"]
    )
