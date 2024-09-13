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


@pytest.mark.parametrize(
    ("model", "msg"),
    [
        ("ScaleStandard", "Cannot standardize 'col' - the standard deviation is zero"),
        (
            "ScaleMinMax",
            "Cannot standardize 'col' - the maximum and minimum values are equal",
        ),
    ],
)
def test_scale_unique_col(model, msg):
    table = ibis.memtable({"col": [1]})
    scale_class = getattr(ml, model)
    step = scale_class("col")
    with pytest.raises(ValueError, match=msg):
        step.fit_table(table, ml.core.Metadata())
