import ibis
import ibis.expr.datatypes as dt
from ibis import _

import ibis_ml as ml

from ibis.backends import _get_backend_names

def test_backends():
    t = ibis.memtable({"col": ["a"]})
    backends = _get_backend_names()
    print(backends)
    for b in backends:
        print(b)
        if b in ("duckdb", "bigquery"):
            print(ibis.to_sql(t, dialect=b))
    step = ml.OneHotEncode(ml.string())
    step.fit_table(t, ml.core.Metadata())
    step.transform_table(t)
    print(ibis.to_sql(step))
    assert False
