# IbisML

[![Build status](https://github.com/ibis-project/ibis-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/ibis-project/ibis-ml/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://ibis-project.github.io/ibis-ml/)
[![License](https://img.shields.io/github/license/ibis-project/ibis-ml.svg)](https://github.com/ibis-project/ibis-ml/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ibis-ml.svg)](https://pypi.org/project/ibis-ml/)

## What is IbisML?

IbisML is a library for building scalable ML pipelines using Ibis:

- Preprocess your data at scale on any [Ibis](https://ibis-project.org/)-supported
  backend.
- Compose [`Recipe`](/reference/core.html#ibis_ml.Recipe)s with other scikit-learn
  estimators using
  [`Pipeline`](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators)s.
- Seamlessly integrate with [scikit-learn](https://scikit-learn.org/stable/),
  [XGBoost](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), and
  [PyTorch](https://skorch.readthedocs.io/en/stable/) models.

## How do I install IbisML?

```bash
pip install ibis-ml
```

## How do I use IbisML?

With recipes, you can define sequences of feature engineering steps to get your data
ready for modeling. For example, create a recipe to replace missing values using the
mean of each numeric column and then normalize numeric data to have a standard deviation
of one and a mean of zero.

```python
import ibis_ml as ml

imputer = ml.ImputeMean(ml.numeric())
scaler = ml.ScaleStandard(ml.numeric())
rec = ml.Recipe(imputer, scaler)
```

A recipe can be chained in a
[`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
like any other
[transformer](https://scikit-learn.org/stable/glossary.html#term-transformer).

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline([("rec", rec), ("svc", SVC())])
```

The pipeline can be used as any other estimator and avoids leaking the test set into the
train set.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe.fit(X_train, y_train).score(X_test, y_test)
```
