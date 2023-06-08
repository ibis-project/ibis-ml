import pytest

import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.ScaleStandard(ml.numeric()), "ScaleStandard(numeric())"),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.ScaleStandard({"x": (0.5, 0.2)}), "ScaleStandard<x>"),
    ],
)
def test_transform_repr(transform, sol):
    assert repr(transform) == sol
