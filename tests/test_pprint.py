import pytest

import ibis_ml as ml


@pytest.fixture
def rec():
    imputer = ml.ImputeMean(ml.numeric())
    scaler = ml.ScaleStandard(ml.numeric())
    encoder = ml.OneHotEncode(ml.string(), min_frequency=20, max_categories=10)
    return ml.Recipe(imputer, scaler, encoder)


@pytest.fixture
def pipe(rec):
    pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    return Pipeline([("rec", rec), ("svc", SVC())])


def test_steps(rec):
    expected = [
        "ImputeMean(numeric())",
        "ScaleStandard(numeric())",
        "OneHotEncode(string(), min_frequency=20, max_categories=10)",
    ]
    assert [repr(step) for step in rec.steps] == expected


def test_recipe(rec):
    expected = """
Recipe(ImputeMean(numeric()),
       ScaleStandard(numeric()),
       OneHotEncode(string(), min_frequency=20, max_categories=10))"""

    expected = expected[1:]  # remove first \n
    assert repr(rec) == expected


def test_recipe_in_sklearn_pipeline(pipe):
    expected = """
Pipeline(steps=[('rec',
                 Recipe(ImputeMean(numeric()), ScaleStandard(numeric()),
                        OneHotEncode(string(),
                                     min_frequency=20,
                                     max_categories=10))),
                ('svc', SVC())])"""

    expected = expected[1:]  # remove first \n
    assert repr(pipe) == expected
