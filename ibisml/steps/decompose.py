from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import sklearn.decomposition

from ibisml.core import Metadata, Step
from ibisml.select import SelectionType, selector

if TYPE_CHECKING:
    import ibis.expr.types as ir


class PCA(Step):
    """A step for principal component analysis (PCA).

    Parameters
    ----------
    inputs
        A selection of columns to transform.
    n_components
        Number of components to keep.
    """

    def __init__(
        self, inputs: SelectionType, n_components: int | float | str | None = None
    ):
        self.inputs = selector(inputs)
        self.n_components = n_components

    def _repr(self) -> Iterable[tuple[str, Any]]:
        yield ("", self.inputs)
        yield ("", self.n_components)

    def fit_table(self, table: ir.Table, metadata: Metadata) -> None:
        PerformanceWarning = RuntimeWarning  # TODO(deepyaman): Define a custom warning.
        warnings.warn(
            f"{type(self)} cannot be fit natively; falling back to scikit-learn.",
            PerformanceWarning,
            stacklevel=2,
        )
        columns = self.inputs.select_columns(table, metadata)
        X = table[columns].to_pandas()  # TODO(deepyaman): Handle empty selection given.
        pca = sklearn.decomposition.PCA(n_components=self.n_components).fit(X)

        # TODO(deepyaman): Consider defining list of attributes to copy.
        self.components_ = pca.components_
        self.explained_variance_ = pca.explained_variance_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.singular_values_ = pca.singular_values_
        self.mean_ = pca.mean_
        self.n_components_ = pca.n_components_
        self.n_samples_ = pca.n_samples_
        self.noise_variance_ = pca.noise_variance_
        self.n_features_in_ = pca.n_features_in_
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

    def transform_table(self, table: ir.Table) -> ir.Table:
        X = table[self.feature_names_in_]

        # X = X - self.mean_
        X = X[
            (
                (X[c] - self.mean_[i]).name(c)
                for i, c in enumerate(self.feature_names_in_)
            )
        ]

        # X_transformed = X @ self.components_.T
        def _dot_product(X, v):
            return sum(X[i] * vi for i, vi in enumerate(v))

        X_transformed = X[
            (
                _dot_product(X, self.components_[k]).name(f"PC{k + 1}")
                for k in range(self.n_components_)
            )
        ]

        return X_transformed
