from __future__ import annotations

from collections.abc import Sequence

import ibis.expr.types as ir


class Transform:
    def transform(self, table: ir.Table) -> ir.Table:
        raise NotImplementedError


class Step:
    def fit(self, table: ir.Table, outcomes: list[str]) -> Transform:
        raise NotImplementedError


class Recipe:
    steps: list[Step | Transform]
    transforms: list[Transform] | None

    def __init__(self, steps: Sequence[Step | Transform]):
        self.steps = list(steps)
        self.transforms = None

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.steps)
        return f"Recipe:\n{parts}"

    def fit(
        self, table: ir.Table, outcomes: str | Sequence[str] | None = None
    ) -> RecipeTransform:
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

        return RecipeTransform(transforms)


class RecipeTransform:
    transforms: list[Transform]

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __repr__(self) -> str:
        parts = "\n".join(f"- {s!r}" for s in self.transforms)
        return f"RecipeTransform:\n{parts}"

    def transform(self, table: ir.Table) -> ir.Table:
        for transform in self.transforms:
            table = transform.transform(table)
        return table
