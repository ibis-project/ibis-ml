# Objective

This document lays out some potential directions for IbisML. Ideas are not necessarily fully-formed
or validated (yet). Please feel free to comment or add additional opportunities.

# Current state

IbisML 0.1.2 was released on Jul 24, 2024 with the following main functionalities:

- A set of [commonly-used ML preprocessing transformers](https://github.com/ibis-project/ibis-ml/issues/32)
  that can be fit and applied in SQL (using Ibis)
- The ability to compose these transformers into IbisML `Recipe` objects, that can be used within
  scikit-learn pipelines
- Additional utilities, like a reproducible, database-native train-test splitter and an array of
  column selectors
- Examples that illustrate the use of IbisML with various ML modeling frameworks, including
  scikit-learn, XGBoost, and PyTorch

# Opportunities

## Support more parts of the end-to-end ML workflow

![End-to-end model training process](https://github.com/user-attachments/assets/fbb5955b-06a4-4e0c-b76b-ebc1fd83daea)

IbisML primarily supports ML preprocessing steps, which is a very narrow focus. Furthermore, in the
standard ML workflow, the gaps in coverage between Ibis and IbisML mean that the end-to-end story is
disjoint. Specifically, Ibis is a good fit for feature engineering, and IbisML can perform
preprocessing (and, to an extent, train-test splitting); however, IbisML doesn't support CV
splitting (or hyperparameter tuning) workflows that are normally part of the training process. Users
have tried using scikit-learn's CV capabilities, but they aren't integrated with IbisML; for
hyperparameter tuning, we could further investigate and document integration with popular frameworks
such as Optuna.

Benefits:

- Most aligned with current direction of IbisML
- Actual community requests
  - https://github.com/ibis-project/ibis-ml/issues/135
  - https://github.com/ibis-project/ibis-ml/issues/136

Questions:

- Is a database the right place for all of these operations? Is it efficient?

## Leverage Apache Arrow for ML preprocessing

While SQL is supported by all databases, it's limiting (and possibly inefficient) for ML
preprocessing workloads. Apache Arrow is a standard that is being adopted by more and more computing
engines that may be better suited for ML preprocessing; in particular, "if the underlying data is in
the Arrow format, then the ML preprocessing can be extremely efficient since it would enable
avoiding format conversions etc." (DuckDB, DataFusion, Polars, and Theseus, as well as pandas and
Dask, all represent data as Arrow already.)

Doing ML preprocessing via Arrow could mean using Arrow Compute Functions, or by working more with
existing ML frameworks that support Arrow natively (and possibly improving Arrow support in other
frameworks). For example, https://github.com/scikit-learn/scikit-learn/issues/26024 "definitely
helps towards that." (It sounds like https://arrow.apache.org/docs/python/numpy.html#arrow-to-numpy
means you can just create a view with no copy, and possibly pass this to scikit-learn, possibly
[parallelized](https://docs.ray.io/en/latest/ray-more-libs/joblib.html); scikit-learn would also
support a lot of these steps on other backends implementing the Array API, including CuPy, PyTorch
Tensors, and [Dask](https://github.com/ibis-project/ibis/issues/9891)).

Benefits:

- Further push the adoption of Arrow in the ML space
- It's not necessary to do ML preprocessing in SQL, and training somewhere else; preprocessing and
  training could be more effectively grouped together, which is representative of the standard ML
  workflow
- Quite possibly more efficient (to be verified), and definitely more flexible, than doing ML
  preprocessing in SQL

Questions:

- Is there a benefit to implementing ML preprocessing in Arrow Compute, or should we just delegate
  to an existing framework?
- If we delegate to something like scikit-learn, how do we make sure it scales?
  https://github.com/scikit-learn/scikit-learn/issues/22352 suggests some performance increases, but
  that's probably not sufficient at a very large scale. Can something like this be parallelized with
  Ray?
- Ray itself supports ML preprocessing to some extent, and also supports Arrow/Arrow Flight/etc. Why
  not just use Ray?
- Is there an opportunity to more generically provide distributed Arrow-based ML?
  https://blog.lambdaclass.com/ballista-a-distributed-compute-platform-made-with-rust-and-apache-arrow/
  seems to indicate some (future) interest in Apache DataFusion Ballista ML?
- If we go down this route, do we care about being able to do ML preprocessing in SQL anymore (for
  backends that don't support Arrow, and possibly also don't support efficient output to Dask)?

## Build a unified interface for doing ML on the database

[PostgresML](https://github.com/postgresml/postgresml) allows users to do everything from feature
engineering to model training and inference (and, arguably, more in the MLOps space) in Postgres.
[BigQuery ML](https://cloud.google.com/bigquery/docs/bqml-introduction) let’s you do the same in
BigQuery. Both share some common goals:

- Bring the ML to the data
- Require knowledge of fewer frameworks to do ML (e.g. empowering SQL users, like business analysts)

PostgresML is a Rust extension, that relies on other Rust libraries (e.g. XGBoost Rust wrapper) or
Python running via pyo3. BigQuery ML can train models in Vertex AI, or reference models from Hugging
Face. However, a lot of Ibis backends don’t have such functionality. Could we implement similar
functionality for DuckDB (C extension), Theseus, DataFusion, etc., and provide a unifying layer? In
the future, can we bring the kinds of functionality in PostgresML to any Ibis backend?

Benefits:

- Try to reduce data transfer between frameworks

Questions:

- Is this composable (at all)? This might require pushing a lot of work in different backends, and
  what's to guarantee would the resulting implementations be equivalent?
- Should this focus on inference? It seems more likely that a database (or at least UDFs, including
  Arrow UDFs) could do well at this part.
  [There are also some others trying to play in this space.](https://www.letsql.com/posts/builtin-predict-udf/)
- Is the value of requiring less knowledge/fewer frameworks appealing/relevant to our target users?

## Unify existing ML APIs

Just like Ibis provides a unified API to execute the same code across various existing backends,
IbisML could unify across various existing ML APIs. Why? If you're doing distributed training,
you're already using some sort of distributed solution like Ray, Dask, Spark, etc. (e.g. Dask and
Ray are a couple of the most popular ways to do distributed XGBoost). You can also use frameworks on
GPU, like cuML. These frameworks can already handle ML and ML preprocessing at massive scale (e.g.
[tens of petabytes at Pinterest](https://www.youtube.com/watch?v=I1eTzQs9QkU)), so it's not clear
that you'd necessarily want to push the ML preprocessing to the database (where such capabilities
don't natively exist).

That said, there are probably differences in syntax and support across these frameworks (although
they all claim to try and model off scikit-learn API). Can IbisML provide a unified layer so you can
write the same code, and have it work seamlessly on scikit-learn/Ray/Dask/Spark/cuML/etc.?

Benefits:

- No disjoint experience in the logical ML pipeline (between preprocessing and modeling)
- We know that these distributed ML solutions can scale, so we could definitely support large-scale
  use cases

Questions:

- Is there sufficient value in unifying across a lot of APIs that already try to follow the
  scikit-learn "standard"?
