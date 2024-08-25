from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_is_fitted

from .core import normalize_table, Metadata
from .steps import (
    CreatePolynomialFeatures
)


class BaseIbisTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ibis_step, ibis_step_params):
        self.columns = columns
        self.ibis_step = ibis_step
        self.ibis_step_params = ibis_step_params

        for attribute, value in ibis_step_params.items():
            setattr(self, attribute, value)

    def fit(self, X, y=None):
        # TODO 1: This is typically where check_array is called.
        # check_array is responsible for input validation and casting
        # to numpy. Here we would need a casting to ibis so that the
        # following code runs no matter what type of df X has.
        self.ibis_ml_step_ = self.ibis_step(self.columns, **self.ibis_step_params)

        table, targets, index = normalize_table(X, y)
        metadata = Metadata(targets=targets)

        if index is not None:
            table = table.drop(index)

        self.ibis_ml_step_.fit_table(table, metadata)


        # TODO 3: In order to have set_output output the right
        # column name when set e.g. to pandas or polars, we need
        # to follow sklearn's feature names API. In particular,
        # fit should set the following attributes
        # self.n_features_in_ = ...
        # self.feature_names_in_ = ...
        
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self)

        # TODO 1: An ibis-friendly check_array should be called

        table, targets, index = normalize_table(X, maintain_order=True)
        if targets:
            table = table.drop(*targets)

        X_t = self.ibis_ml_step_.transform_table(table)

        if index is not None:
            X_t = X_t.order_by(index).drop(index)

        return X_t
    
    # TODO 2: We should have "ibis" as a transform option
    # This requires to develop and register a custom Adapter, 
    # which is important e.g. to have ibis Transformers compatible
    # with FeatureUnion. For that reason transform's default
    # would have to be "ibis".
    def set_output(self, *, transform="ibis"):
        """Set output container.

        Parameters
        ----------
        transform : {"default", "pandas", "polars", "ibis"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `"polars"`: Polars output
            - `"ibis"`: Ibis output
            - `None`: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        return TransformerMixin.set_output(self, transform)

    # TODO 3: Feature names transformation should be defined
    # in the following function.
    # def get_feature_names_out(self, input_features=None):
    #     pass


class PolynomialFeatures(BaseIbisTransformer):
    def __init__(self, columns=None, degree=2):
        BaseIbisTransformer.__init__(
            self, 
            columns=columns, 
            ibis_step=CreatePolynomialFeatures, 
            ibis_step_params={"degree": degree},
        )
