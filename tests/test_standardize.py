import ibis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis_ml as ml


def test_scalestandard():
    cols = np.arange(0, 100)
    mean = np.mean(cols)
    std = np.std(cols)
    table = ibis.memtable({"col": cols})
    step = ml.ScaleStandard("col")
    step.fit_table(table, ml.core.Metadata())
    result = step.transform_table(table)
    expected = pd.DataFrame({"col": (cols - mean) / std})
    tm.assert_frame_equal(result.execute(), expected, check_exact=False)


def test_scaleminmax():
    cols = np.arange(0, 100)
    min_val = np.min(cols)
    max_val = np.max(cols)
    table = ibis.memtable({"col": cols})
    step = ml.ScaleMinMax("col")
    step.fit_table(table, ml.core.Metadata())
    result = step.transform_table(table)
    expected = pd.DataFrame({"col": (cols - min_val) / (max_val - min_val)})
    tm.assert_frame_equal(result.execute(), expected, check_exact=False)


@pytest.mark.parametrize("scaler", ["ScaleStandard", "ScaleMinMax"])
def test_constant_columns(scaler):
    table = ibis.memtable({"int_col": [100], "float_col": [100.0]})
    scaler_class = getattr(ml, scaler)
    scale_step = scaler_class(ml.numeric())
    scale_step.fit_table(table, ml.core.Metadata())
    result = scale_step.transform_table(table)
    expected = pd.DataFrame({"int_col": [0.0], "float_col": [0.0]})
    tm.assert_frame_equal(result.execute(), expected)
