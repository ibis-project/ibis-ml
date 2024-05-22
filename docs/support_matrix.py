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


def make_support_matrix():
    all_steps = list(get_leaf_classes(ml.Step))
    with Path("./step_config.yml").open() as file:
        step_config = yaml.safe_load(file)

    expanded_steps = []
    for step in all_steps:
        step_name = step.__name__
        default_input = "numeric"
        configurations = step_config.get(step_name, {}).get("configurations", [])

        if configurations:
            for name_config in configurations:
                default_input = name_config["config"].get("inputs", default_input)
                expanded_steps.append(
                    {
                        "name": step.__name__,
                        "step_params": name_config["name"],
                        "category": str(step.__module__).split(".")[-1][1:],
                        "params": {
                            **name_config["config"],
                            "inputs": getattr(ml, default_input)(),
                        },
                    }
                )
        else:
            expanded_steps.append(
                {
                    "name": step.__name__,
                    "step_params": "",
                    "category": str(step.__module__).split(".")[-1][1:],
                    "params": {"inputs": getattr(ml, default_input)()},
                }
            )

    backends = sorted(ep.name for ep in ibis.util.backend_entry_points())
    alltypes = {
        "string_col": np.array(["a", None, "b"], dtype="str"),
        "int_col": np.array([1, 2, 3], dtype="int64"),
        "floating_col": np.array([1.0, 2.0, 3.0], dtype="float64"),
        "date_col": [date(2017, 4, 2), date(2017, 4, 2), date(2017, 4, 2)],
        "time_col": [time(9, 1, 1), time(10, 1, 11), None],
        "datetime_col": [
            datetime(2017, 4, 2, 10, 1, 0),
            datetime(2018, 4, 2, 10, 1, 0),
            None,
        ],
        "target": np.array([1, 0, 1], dtype="int8"),
    }
    no_time = {
        "string_col": np.array(["a", None, "b"], dtype="str"),
        "int_col": np.array([1, 2, 3], dtype="int64"),
        "floating_col": np.array([1.0, None, np.nan], dtype="float64"),
        "date_col": [date(2017, 4, 2), date(2017, 4, 2), date(2017, 4, 2)],
        "datetime_col": [
            datetime(2017, 4, 2, 10, 1, 0),
            datetime(2018, 4, 2, 10, 1, 0),
            None,
        ],
        "target": np.array([1, 0, 1], dtype="int8"),
    }

    steps = {"steps": expanded_steps}

    for backend in backends:
        results = []
        for step_dict in expanded_steps:
            step_name = step_dict["name"]
            if step_name in ("Drop", "Cast", "MutateAt", "Mutate"):
                step_dict["step_params"] = "operation-specific"
                results.append(True)
                continue
            if backend in ["pandas", "polars", "dask"]:
                # Dataframe backend does not have to_sql()
                # dask does not support quantile
                if backend == "dask" and step_dict["step_params"] == "quantile":
                    results.append(False)
                else:
                    results.append(True)
                continue

            if backend == "druid":
                # Druid does not support time type
                data = ibis.memtable(no_time)
                if step_name == "ExpandTime":
                    results.append(False)
                    continue
            else:
                data = ibis.memtable(alltypes)

            # construct a step
            step = getattr(ml, step_dict["name"])(**step_dict["params"])
            metadata = ml.core.Metadata(targets=("target",))
            step.fit_table(data, metadata)
            output = step.transform_table(data)

            try:
                if hasattr(step, "_fit_expr"):
                    for expr in step._fit_expr:  # noqa: SLF001
                        ibis.to_sql(expr, backend)
                ibis.to_sql(output, backend)
                results.append(True)
            except ibis.common.exceptions.TranslationError:
                results.append(False)
            except AttributeError:
                # clickhouse does not support scale = 3 for ExtractMillisecond
                results.append(False)
        steps[backend] = list(results)

    support_matrix = (
        pd.DataFrame(steps)
        .assign(
            Category=lambda df: df["steps"].apply(lambda x: x["category"]),
            Step=lambda df: df["steps"].apply(lambda x: x["name"]),
            Param=lambda df: df["steps"].apply(lambda x: x["step_params"]),
        )
        .drop(["steps"], axis=1)
        .set_index(["Category", "Step", "Param"])
        .sort_index()
    )

    all_visible_ops_count = len(support_matrix)
    coverage = pd.Index(
        support_matrix.sum()
        .map(lambda n: f"{n} ({round(100 * n / all_visible_ops_count)}%)")
        .T
    )
    support_matrix.columns = pd.MultiIndex.from_tuples(
        list(zip(support_matrix.columns, coverage)), names=("Backend", "Step coverage")
    )

    return support_matrix


if __name__ == "__main__":
    print(make_support_matrix())  # noqa: T201
