# IbisML

[![Build status](https://github.com/ibis-project/ibisml/actions/workflows/ci.yml/badge.svg)](https://github.com/ibis-project/ibisml/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://ibis-project.github.io/ibisml/)
[![License](https://img.shields.io/github/license/ibis-project/ibisml.svg)](https://github.com/ibis-project/ibisml/blob/main/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/ibisml.svg)](https://pypi.org/project/ibisml/)

`ibisml` is a _work-in-progress_ library for developing Machine Learning
feature engineering pipelines using [ibis](https://ibis-project.org/). These
pipelines can then be used to transform and feed data to other machine learning
libraries like [xgboost](https://xgboost.readthedocs.io) or
[scikit-learn](https://scikit-learn.org).

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

By using `ibis` for preprocessing and feature engineering, feature engineering
pipelines may be compiled to SQL and executed on a wide range of [performant
and scalable backends](https://ibis-project.org/support_matrix). No more need
to rewrite code for production deployments, pipelines may be developed locally
(against e.g. `duckdb`) and deployed to production (against e.g. `spark`) with
only a single line of code change.

## Help Wanted!

`ibisml` is a work-in-progress. If you're interested in getting involved
(whether through feature requests, PRs, or just sharing opinions), we'd love to
hear from you.
