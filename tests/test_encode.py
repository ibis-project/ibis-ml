import pytest

import ibisml as ml


@pytest.mark.parametrize(
    "step, sol",
    [
        (ml.OneHotEncode(ml.string()), "OneHotEncode(string())"),
        (
            ml.OneHotEncode(ml.string(), min_frequency=0.1),
            "OneHotEncode(string(), min_frequency=0.1)",
        ),
        (
            ml.OneHotEncode(ml.string(), max_categories=10),
            "OneHotEncode(string(), max_categories=10)",
        ),
        (ml.CategoricalEncode(ml.string()), "CategoricalEncode(string())"),
        (
            ml.CategoricalEncode(ml.string(), min_frequency=0.1),
            "CategoricalEncode(string(), min_frequency=0.1)",
        ),
        (
            ml.CategoricalEncode(ml.string(), max_categories=10),
            "CategoricalEncode(string(), max_categories=10)",
        ),
    ],
)
def test_step_repr(step, sol):
    assert repr(step) == sol


@pytest.mark.parametrize(
    "transform, sol",
    [
        (ml.transforms.OneHotEncode({"x": ["a", "b"]}), "OneHotEncode<x>"),
        (ml.transforms.CategoricalEncode({"x": ["a", "b"]}), "CategoricalEncode<x>"),
    ],
)
def test_transform_repr(transform, sol):
    assert repr(transform) == sol
