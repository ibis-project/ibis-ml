from __future__ import annotations

from collections import defaultdict

import ibis
import ibis.expr.types as ir

import ibisml as ml
from ibisml.core import Step, Transform
from ibisml.select import SelectionType, selector


class OneHotEncode(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def fit(self, table: ir.Table, outcomes: list[str]) -> Transform:
        columns = (self.inputs - outcomes).select_columns(table)

        # We execute once for each type kind in the inputs. In the common case
        # (only string inputs) this means a single execution even for multiple
        # columns.
        groups = defaultdict(list)
        for c in columns:
            groups[table[c].type()].append(c)

        categories = {}
        for group_type, group_cols in groups.items():
            # Results in a frame like:
            # value | column
            # --------------
            # A     | col1
            # B     | col1
            # X     | col2
            # ...   | ...
            query = ibis.union(
                *(
                    (
                        table.select(value=col, column=ibis.literal(col))
                        .distinct()
                        .order_by("value")
                    )
                    for col in group_cols
                )
            )
            result_groups = query.execute().groupby("column")

            for col in group_cols:
                categories[col] = result_groups.get_group(col)["value"].to_list()

        return ml.transforms.OneHotEncode(categories)
