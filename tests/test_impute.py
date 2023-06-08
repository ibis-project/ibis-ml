import pytest

import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.FillNA(ml.numeric(), 0), "FillNA(numeric(), 0)"),
        (ml.ImputeMean(ml.everything()), "ImputeMean(everything())"),
        (ml.ImputeMode(ml.everything()), "ImputeMode(everything())"),
        (ml.ImputeMedian(ml.everything()), "ImputeMedian(everything())"),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.FillNA({"x": 1, "y": 2}), "FillNA<x, y>"),
    ],
)
def test_transform_repr(transform, sol):
    assert repr(transform) == sol
