# Ibis-ML

`ibisml` is a *work-in-progress* library for developing Machine Learning
feature engineering pipelines using [ibis](https://ibis-project.org/). These
pipelines can then be used to transform and feed data to other machine learning
libraries like [xgboost](https://xgboost.readthedocs.io) or
[scikit-learn](https://scikit-learn.org).

```python
import ibis
import ibisml as ml

# Load some training and testing data
train = ibis.read_csv("training.csv")
test = ibis.read_csv("testing.csv")

# A recipe for a feature engineering pipeline that:
# - imputes missing values in numeric columns with their mean
# - applies standard scaling to all numeric columns
# - one-hot-encodes all nominal columns
recipe = ml.Recipe(
    ml.ImputeMean(ml.numeric()),
    ml.ScaleStandard(ml.numeric()),
    ml.OneHotEncode(ml.nominal()),
)

# Fit the recipe against the training data
transform = recipe.fit(train, outcomes=["outcome_col"])

# Transform the training data and train a scikit-learn model
from sklearn.svm import LinearSVC
model = LinearSVC()

df_train = transform(train).to_pandas()
X = df_train[transform.features]
y = df_train[transform.outcomes]
model.fit(X, y)

# Transform the testing data and use the model to predict results
df_test = transform(test).to_pandas()
X = df_test[transform.features]
y = df_test[transform.outcomes]
y_pred = model.predict(X)
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
