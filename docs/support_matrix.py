from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import yaml

import ibis_ml as ml


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        if not child_class.__subclasses__():
            yield child_class
        else:
            yield from get_leaf_classes(child_class)


def check_backend(backend, exprs):
    if backend in ("pandas", "polars", "dask"):
        try:
            con = ibis.connect(f"{backend}://")
            for expr in exprs:
                con.execute(expr)
            return True
        except ibis.common.exceptions.IbisError:
            return False
    else:
        try:
            for expr in exprs:
                ibis.to_sql(expr, backend)
            return True
        except ibis.common.exceptions.TranslationError:
            return False
        except AttributeError:
            return False


def make_support_matrix():
    all_steps = list(get_leaf_classes(ml.Step))
    with Path("./step_config.yml").open() as file:
        step_config = yaml.safe_load(file)

    expanded_steps = []
    for step in all_steps:
        step_name = step.__name__
        module_category = str(step.__module__).split(".")[-1][1:]
        configurations = step_config.get(step_name, {}).get("configurations", [])

        if configurations:
            expanded_steps.append(
                [
                    {
                        "name": step_name,
                        "step_params": config["name"],
                        "category": module_category,
                        "params": {
                            **config["config"],
                            "inputs": getattr(
                                ml, config["config"].get("inputs", "numeric")
                            )(),
                        },
                    }
                    for config in configurations
                ]
            )
        else:
            expanded_steps.append(
                [
                    {
                        "name": step_name,
                        "step_params": "None",
                        "category": module_category,
                        "params": {"inputs": ml.numeric()},
                    }
                ]
            )

    backends = sorted(ep.name for ep in ibis.util.backend_entry_points())
    alltypes = {
        "string": np.array(["a", None, "b"], dtype="str"),
        "int": np.array([1, 2, 3], dtype="int64"),
        "floating": np.array([1.0, 2.0, 3.0], dtype="float64"),
        "date": [date(2017, 4, 2), date(2017, 4, 2), date(2017, 4, 2)],
        "time": [time(9, 1, 1), time(10, 1, 11), None],
        "datetime": [
            datetime(2017, 4, 2, 10, 1, 0),
            datetime(2018, 4, 2, 10, 1, 0),
            None,
        ],
        "target": np.array([1, 0, 1], dtype="int8"),
    }

    steps = {"steps": expanded_steps}
    unsupported_cols = {"druid": ["time"]}

    backend_specific = {
        "support": ["backend-specific"],
        "not_support": ["backend-specific"],
    }
    special_step = {
        "Drop": {"support": [], "not_support": []},
        "Cast": backend_specific,
        "MutateAt": backend_specific,
        "Mutate": backend_specific,
    }

    for backend in backends:
        results = []
        for expand_step in expanded_steps:
            res = {"support": [], "not_support": []}
            for step_dict in expand_step:
                step_name = step_dict["name"]
                input_type = type(step_dict["params"]["inputs"]).__name__
                if step_name in special_step:
                    res = special_step[step_name]
                    continue

                if input_type in unsupported_cols.get(backend, []):
                    res["not_support"].append(step_dict["step_params"])
                    continue

                df = pd.DataFrame(alltypes).drop(
                    columns=unsupported_cols.get(backend, [])
                )
                data = ibis.memtable(df)

                # construct a step
                step = getattr(ml, step_dict["name"])(**step_dict["params"])
                metadata = ml.core.Metadata(targets=("target",))
                step.fit_table(data, metadata)
                output = step.transform_table(data)
                all_expr = []
                if hasattr(step, "_fit_expr"):
                    all_expr.extend(step._fit_expr)  # noqa: SLF001
                all_expr.append(output)

                if check_backend(backend, all_expr):
                    res["support"].append(step_dict["step_params"])
                else:
                    res["not_support"].append(step_dict["step_params"])

            if not res["not_support"]:
                results.append(True)
            elif res["not_support"] and not res["support"]:
                results.append(False)
            else:
                results.append(",".join(set(res["support"])))

        steps[backend] = list(results)

    support_matrix = (
        pd.DataFrame(steps)
        .assign(
            Category=lambda df: df["steps"].apply(lambda x: x[0]["category"]),
            Step=lambda df: df["steps"].apply(lambda x: x[0]["name"]),
        )
        .drop(["steps"], axis=1)
        .set_index(["Category", "Step"])
        .sort_index()
    )

    def count_full(column):
        return sum(1 for value in column if isinstance(value, bool))

    all_visible_ops_count = len(support_matrix)
    fully_coverage = pd.Index(
        support_matrix.apply(count_full)
        .map(lambda n: f"{n} ({round(100 * n / all_visible_ops_count)}%)")
        .T
    )

    def count_partial(column):
        return sum(1 for value in column if isinstance(value, str))

    partial_coverage = pd.Index(
        support_matrix.apply(count_partial)
        .map(lambda n: f"{n} ({round(100 * n / all_visible_ops_count)}%)")
        .T
    )
    support_matrix.columns = pd.MultiIndex.from_tuples(
        list(zip(support_matrix.columns, fully_coverage, partial_coverage)),
        names=("Backend", "Full coverage", "Partial coverage"),
    )

    return support_matrix


if __name__ == "__main__":
    print(make_support_matrix())  # noqa: T201
