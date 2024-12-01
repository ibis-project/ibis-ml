from unittest.mock import patch

import ibis
import ibis.expr.types as ir
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from ibis import _

import ibis_ml as ml
from ibis_ml.core import normalize_table


class Shuffle(ml.Step):
    """A Step to reorder the table, for use in testing that order is maintained"""

    def is_fitted(self):
        return True

    def transform_table(self, table):
        return table.order_by(ibis.random())


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 0, 1, 0, 1],
            "c": ["x", "x", "y", "x", "y"],
            "y": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def table(df):
    return ibis.memtable(df)


def test_is_fitted(table):
    r = ml.Recipe(ml.Drop(~ml.numeric()))
    assert not r.is_fitted()

    r.fit(table, "y")
    assert r.is_fitted()


def test_sklearn_clone(table):
    sklearn = pytest.importorskip("sklearn")

    r1 = ml.Recipe(ml.Drop(~ml.numeric()))
    assert not r1.is_fitted()

    r1.fit(table, "y")
    assert r1.is_fitted()

    r2 = sklearn.clone(r1)
    assert not r2.is_fitted()

    r2.fit(table, "y")
    assert r1.to_ibis(table).equals(r2.to_ibis(table))


def test_get_visual_block_recipe():
    pytest.importorskip("sklearn")
    from sklearn.utils._estimator_html_repr import _get_visual_block

    rec = ml.Recipe(ml.ImputeMean(ml.numeric()), ml.ScaleStandard(ml.numeric()))
    est_html_info = _get_visual_block(rec)
    assert est_html_info.kind == "serial"
    assert est_html_info.estimators == rec.steps
    assert est_html_info.names == [
        "imputemean: ImputeMean",
        "scalestandard: ScaleStandard",
    ]
    assert est_html_info.name_details == [str(est) for est in rec.steps]


@pytest.mark.parametrize("include_y", [False, True])
def test_in_memory_workflow(df, include_y):
    X = df[["a", "b", "c"]]

    r = ml.Recipe(ml.Mutate(d=_.a + _.b), ml.Drop(~ml.numeric()))

    if include_y:
        r.fit(X, y=df.y)
    else:
        r.fit(X)
    assert r.is_fitted()
    res = r.transform(X)

    assert isinstance(res, np.ndarray)
    sol = X.assign(d=X.a + X.b).drop("c", axis=1).to_numpy()
    np.testing.assert_array_equal(res, sol)


def test_set_output():
    recipe = ml.Recipe(ml.Drop(~ml.numeric()))
    assert recipe.output_format == "default"

    for format in ["default", "pandas", "pyarrow", "polars"]:
        assert recipe.set_output(transform=format) is recipe
        assert recipe.output_format == format

    recipe.set_output(transform="polars")  # something non-standard
    recipe.set_output(transform=None)  # None -> leave unchanged
    assert recipe.output_format == "polars"

    with pytest.raises(
        ValueError, match=r"`transform` must be one of \(.*\), got 'unsupported'"
    ):
        recipe.set_output(transform="unsupported")


@pytest.mark.parametrize("format", ["pandas", "polars", "pyarrow", "default"])
@pytest.mark.parametrize("fit_transform", [False, True])
@pytest.mark.parametrize("include_y", [False, True])
def test_output_formats(table, format, fit_transform, include_y):
    X = table[["a", "b", "c"]]
    y = table["y"] if include_y else None

    if format == "pandas":
        typ = pd.DataFrame
    elif format == "pyarrow":
        typ = pa.Table
    elif format == "polars":
        typ = pytest.importorskip("polars").DataFrame
    else:
        typ = np.ndarray

    r = ml.Recipe(ml.Mutate(d=_.a + _.b), ml.Drop(~ml.numeric()))
    r.set_output(transform=format)
    out = r.fit_transform(X, y=y) if fit_transform else r.fit(X, y=y).transform(X)
    assert isinstance(out, typ)

    out2 = r.fit_transform(table)
    assert isinstance(out2, typ)


def test_to_numpy_errors_non_numeric(table):
    r = ml.Recipe(ml.Mutate(d=_.a + _.b))
    r.fit(table)
    with pytest.raises(ValueError, match="Not all output columns are numeric"):
        r.to_numpy(table)


@pytest.mark.parametrize(
    "format", ["numpy", "pandas", "pyarrow", "polars", "ibis-table"]
)
def test_input_formats(format):
    r = ml.Recipe(ml.Cast(ml.everything(), "float64"))
    X = np.eye(3, dtype="i8")
    if format == "polars":
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(X, schema=["x0", "x1", "x2"])
    elif format != "numpy":
        X = pd.DataFrame(X, columns=["x0", "x1", "x2"])
        if format == "ibis-table":
            X = ibis.memtable(X)
        elif format == "pyarrow":
            X = pa.Table.from_pandas(X)
    r.fit(X)
    out = r.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == "f8"


@pytest.mark.parametrize(
    "format", ["numpy", "pandas", "pyarrow", "polars", "ibis-table"]
)
def test_transform_in_memory_data_maintains_order(format):
    r = ml.Recipe(ml.Cast(ml.everything(), "float64"), Shuffle())

    X = np.vstack([np.arange(100), np.arange(100, 200)]).T
    if format == "polars":
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(X, schema=["x0", "x1"])
    elif format != "numpy":
        X = pd.DataFrame(X, columns=["x0", "x1"])
        if format == "ibis-table":
            X = ibis.memtable(X)
        elif format == "pyarrow":
            X = pa.Table.from_pandas(X)
    r.fit(X)
    out = r.to_pandas(X)

    # table inputs won't maintain order, in-memory inputs will
    should_be_ordered = format != "ibis-table"

    assert list(out.columns) == ["x0", "x1"]

    for col in out.columns:
        assert out[col].dtype == "f8"
        assert out[col].is_monotonic_increasing == should_be_ordered


def test_can_use_in_sklearn_pipeline():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    X = np.array([[1, 3], [2, 4], [3, 5]])
    y = np.array([10, 11, 12])

    r = ml.Recipe(ml.Mutate(x2=_.x0 + _.x1), ml.ScaleStandard(ml.everything()))
    p = Pipeline([("recipe", r), ("model", LinearRegression())])

    # get/set params works
    params = p.get_params()
    p.set_params(**params | {"recipe__scalestandard__inputs": ml.numeric()})
    assert p["recipe"].steps[1].inputs == ml.numeric()

    # fit and predict work
    p.fit(X, y)
    assert isinstance(p.predict(X), np.ndarray)

    # clone works
    p2 = sklearn.clone(p)
    r2 = p2.named_steps["recipe"]
    assert r2 is not r
    assert not r2.is_fitted()


@pytest.mark.parametrize(
    "get_Xy",
    [
        pytest.param(lambda t: (t[["a", "b", "c"]], None), id="Project-None"),
        pytest.param(lambda t: (t[["a", "b", "c"]], t["y"]), id="Project-column"),
        pytest.param(lambda t: (t[["a", "b", "c"]], t[["y"]]), id="Project-table"),
        pytest.param(lambda t: (t.drop("y"), None), id="DropColumns-None"),
        pytest.param(lambda t: (t.drop("y"), t["y"]), id="DropColumns-column"),
        pytest.param(lambda t: (t.drop("y"), t[["y"]]), id="DropColumns-table"),
        pytest.param(lambda t: (t, "y"), id="col-name"),
        pytest.param(lambda t: (t, ["y"]), id="col-names"),
    ],
)
@pytest.mark.parametrize("maintain_order", [False, True])
def test_normalize_table_ibis(table, get_Xy, maintain_order):
    X, y = get_Xy(table)
    t, targets, index = normalize_table(X, y, maintain_order=maintain_order)
    sol_targets = ("y",) if y is not None else ()
    sol_cols = ("a", "b", "c", *sol_targets)
    assert isinstance(t, ir.Table)
    assert tuple(t.columns) == sol_cols
    assert targets == sol_targets
    assert index is None


def test_normalize_table_ibis_errors(table):
    with pytest.raises(TypeError, match="must also be an ibis Table or Column"):
        normalize_table(table, object())

    with pytest.raises(ValueError, match="must not share column names"):
        normalize_table(table, table[["y"]])

    y = ibis.table({"target": "int"}, name="y")
    with pytest.raises(ValueError, match="must directly share a common parent"):
        normalize_table(table, y)


@pytest.mark.parametrize(
    "get_y",
    [
        pytest.param(lambda t: None, id="None"),
        pytest.param(lambda t: t["y"], id="series"),
        pytest.param(lambda t: t[["y"]], id="dataframe"),
        pytest.param(lambda t: t["y"].to_numpy(), id="numpy-1d"),
        pytest.param(lambda t: t[["y"]].to_numpy(), id="numpy-2d"),
    ],
)
@pytest.mark.parametrize("x_is_numpy", [False, True])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_normalize_table_pandas_numpy(df, get_y, x_is_numpy, maintain_order):
    y = get_y(df)
    X = df[["a", "b"]]
    if x_is_numpy:
        X = X.to_numpy()

    t, targets, index = normalize_table(X, y, maintain_order=maintain_order)
    if y is None:
        sol_targets = ()
    elif isinstance(y, np.ndarray):
        sol_targets = ("y",) if y.ndim == 1 else ("y0",)
    else:
        sol_targets = ("y",)

    sol_cols = (("x0", "x1") if x_is_numpy else ("a", "b")) + sol_targets

    if maintain_order:
        sol_cols += (index,)
    assert isinstance(t, ir.Table)
    assert tuple(t.columns) == sol_cols
    assert targets == sol_targets
    assert bool(index is not None) == maintain_order


@pytest.mark.parametrize("y_kind", ["none", "array", "chunked-array", "table"])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_normalize_table_pyarrow(df, y_kind, maintain_order):
    X = pa.Table.from_pydict({"a": [1, 2], "b": [3, 4]})
    if y_kind == "array":
        y = pa.array([5, 6])
    elif y_kind == "chunked-array":
        y = pa.chunked_array([[5, 6]])
    elif y_kind == "table":
        y = pa.Table.from_pydict({"y": [5, 6]})
    else:
        y = None

    t, targets, index = normalize_table(X, y, maintain_order=maintain_order)

    sol_targets = () if y is None else ("y",)
    sol_cols = tuple(X.column_names) + sol_targets
    if maintain_order:
        sol_cols += (index,)

    assert isinstance(t, ir.Table)
    assert tuple(t.columns) == sol_cols
    assert targets == sol_targets
    assert bool(index is not None) == maintain_order


def test_normalize_table_pyarrow_errors():
    X = pa.Table.from_pydict({"x": [1, 2]})
    with pytest.raises(TypeError, match="must also be a pyarrow"):
        normalize_table(X, object())


@pytest.mark.parametrize("y_kind", ["none", "series", "dataframe"])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_normalize_table_polars(df, y_kind, maintain_order):
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    if y_kind == "series":
        y = pl.Series(name="y", values=[5, 6])
    elif y_kind == "dataframe":
        y = pl.DataFrame({"y": [5, 6]})
    else:
        y = None

    t, targets, index = normalize_table(X, y, maintain_order=maintain_order)

    sol_targets = () if y is None else ("y",)
    sol_cols = tuple(X.columns) + sol_targets
    if maintain_order:
        sol_cols += (index,)

    assert isinstance(t, ir.Table)
    assert tuple(t.columns) == sol_cols
    assert targets == sol_targets
    assert bool(index is not None) == maintain_order


def test_normalize_table_polars_errors():
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="must also be a polars"):
        normalize_table(X, object())


@pytest.mark.parametrize("method", ["transform", "to_ibis", "to_pandas", "to_numpy"])
def test_errors_nicely_if_not_fitted(table, method):
    r = ml.Recipe(ml.Drop(~ml.numeric()), ml.ScaleStandard(ml.numeric()))
    with pytest.raises(ValueError, match="not fitted"):
        getattr(r, method)(table)


def test_get_params():
    rec = ml.Recipe(ml.ExpandTimestamp(ml.timestamp()))

    assert "expandtimestamp__components" in rec.get_params(deep=True)
    assert "expandtimestamp__components" not in rec.get_params(deep=False)


def test_set_params():
    rec = ml.Recipe(ml.ExpandTimestamp(ml.timestamp()))

    # Nonexistent parameter in step
    with pytest.raises(
        ValueError,
        match="Invalid parameter 'nonexistent_param' for step ExpandTimestamp",
    ):
        rec.set_params(expandtimestamp__nonexistent_param=True)

    # Nonexistent parameter of pipeline
    with pytest.raises(
        ValueError, match="Invalid parameter 'expanddatetime' for recipe Recipe"
    ):
        rec.set_params(expanddatetime__nonexistent_param=True)


def test_set_params_passes_all_parameters():
    # Make sure all parameters are passed together to set_params
    # of nested estimator.
    rec = ml.Recipe(ml.ExpandTimestamp(ml.timestamp()))
    with patch.object(ml.ExpandTimestamp, "_set_params") as mock_set_params:
        rec.set_params(
            expandtimestamp__inputs=["x", "y"],
            expandtimestamp__components=["day", "year", "hour"],
        )

    mock_set_params.assert_called_once_with(
        inputs=["x", "y"], components=["day", "year", "hour"]
    )


def test_set_params_updates_valid_params():
    # Check that set_params tries to set `replacement_mutateat.inputs`, not
    # `original_mutateat.inputs`.
    original_mutateat = ml.MutateAt("dep_time", ibis._.hour() * 60 + ibis._.minute())
    rec = ml.Recipe(
        original_mutateat, ml.MutateAt(ml.timestamp(), ibis._.epoch_seconds())
    )
    replacement_mutateat = ml.MutateAt("arr_time", ibis._.hour() * 60 + ibis._.minute())
    rec.set_params(
        **{"mutateat-1": replacement_mutateat, "mutateat-1__inputs": ml.cols("arrival")}
    )
    assert original_mutateat.inputs == ml.cols("dep_time")
    assert replacement_mutateat.inputs == ml.cols("arrival")
    assert rec.steps[0] is replacement_mutateat


@pytest.mark.parametrize(
    ("step", "url"),
    [
        (
            ml.Drop(~ml.numeric()),
            "https://ibis-project.github.io/ibis-ml/reference/steps-other.html#ibis_ml.Drop",
        ),
        (Shuffle(), ""),
    ],
)
def test_get_doc_link(step, url):
    assert step._get_doc_link() == url  # noqa: SLF001
