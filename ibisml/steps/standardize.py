from __future__ import annotations

import ibisml as ml
from ibisml.core import Step, Transform
from ibisml.select import SelectionType, selector

import ibis.expr.types as ir

__all__ = ("ScaleStandard",)


class ScaleStandard(Step):
    def __init__(self, inputs: SelectionType):
        self.inputs = selector(inputs)

    def fit(self, table: ir.Table, outcomes: list[str]) -> Transform:
        columns = (self.inputs - outcomes).select_columns(table)

        stats = {}
        if columns:
            aggs = []
            for name in columns:
                c = table[name]
                if not isinstance(c, ir.NumericColumn):
                    raise ValueError(
                        f"Cannot standardize {name!r} - this column is not numeric"
                    )

                aggs.append(c.mean().name(f"{name}_mean"))
                aggs.append(c.std(how="pop").name(f"{name}_std"))

            results = table.aggregate(aggs).execute().to_dict("records")[0]
            for name in columns:
                stats[name] = (results[f"{name}_mean"], results[f"{name}_std"])
        return ml.transforms.ScaleStandard(stats)
