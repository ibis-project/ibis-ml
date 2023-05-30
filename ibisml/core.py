from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Union

import ibis.expr.types as ir

from . import select


InputsType = Union[str, Sequence[str], select.Selector, Callable[[ir.Column], bool]]


def normalize_inputs(inputs: InputsType):
    if isinstance(inputs, str):
        return select.cols(inputs)
    elif isinstance(inputs, (list, tuple)):
        return select.cols(*inputs)
    elif callable(inputs):
        return select.where(inputs)
    elif isinstance(inputs, select.Selector):
        return inputs
    else:
        raise TypeError("inputs must be a str, list of strings, callable, or selector")


class Transform:
    def transform(self, table: ir.Table) -> ir.Table:
        ...


class Step:
    def fit(self, table: ir.Table, outcomes: list[str]) -> Transform:
        ...


class Recipe:
    steps: list[Step | Transform]
    transforms: list[Transform] | None

    def __init__(self, steps: Sequence[Step | Transform]):
        self.steps = list(steps)
        self.transforms = None

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.steps)
        return f"Pipeline:\n{parts}"

    def fit(self, table: ir.Table, outcomes: str | Sequence[str] | None = None) -> None:
        if outcomes is None:
            outcomes = []
        elif isinstance(outcomes, str):
            outcomes = [outcomes]
        else:
            outcomes = list(outcomes)

        transforms = []
        for step in self.steps:
            if isinstance(step, Step):
                transform = step.fit(table, outcomes)
            else:
                transform = step

            transforms.append(transform)
            table = transform.transform(table)

        self.transforms = transforms

    def transform(self, table: ir.Table) -> ir.Table:
        if self.transforms is None:
            raise ValueError(
                "This recipe hasn't been fit - please call `recipe.fit` first"
            )
        for transform in self.transforms:
            table = transform.transform(table)
        return table
