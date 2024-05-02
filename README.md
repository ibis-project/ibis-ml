# IbisML

[![Build status](https://github.com/ibis-project/ibis-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/ibis-project/ibis-ml/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://ibis-project.github.io/ibis-ml/)
[![License](https://img.shields.io/github/license/ibis-project/ibis-ml.svg)](https://github.com/ibis-project/ibis-ml/blob/main/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/ibisml.svg)](https://pypi.org/project/ibisml/)

## What is IbisML?

IbisML is a library for building scalable ML pipelines using Ibis:

- Preprocess your data at scale on any [Ibis](https://ibis-project.org/)-supported
  backend.
- Compose [`Recipe`](/reference/core.html#ibisml.Recipe)s with other scikit-learn
  estimators using
  [`Pipeline`](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators)s.
- Seamlessly integrate with [scikit-learn](https://scikit-learn.org/stable/),
  [XGBoost](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), and
  [PyTorch](https://skorch.readthedocs.io/en/stable/) models.

```python
import ibis
import ibisml as ml

# A recipe for a feature engineering pipeline that:
# - imputes missing values in numeric columns with their mean
# - applies standard scaling to all numeric columns
# - one-hot-encodes all nominal columns
recipe = ml.Recipe(
    ml.ImputeMean(ml.numeric()),
    ml.ScaleStandard(ml.numeric()),
    ml.OneHotEncode(ml.nominal()),
)

# Use the recipe inside of a larger Scikit-Learn pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([("recipe", recipe), ("model", LinearSVC())])

# Fit the recipe against some local training data,
# just as you would with any other scikit-learn model
X, y = load_training_data()
pipeline.fit(X, y)

# Evaluate the model against some local testing data.
X_test, y_test = load_testing_data()
pipeline.score(X_test, y_test)

# Now apply the same preprocessing pipeline against any of ibis's
# supported backends
con = ibis.connect(...)
X_remote = con.table["mytable"]
for batch in recipe.to_pyarrow_batches(X_remote):
    ...
```
