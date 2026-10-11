"""query_data filter_summary: the expression filters rows and can do nothing else.

df.query on pandas' python engine ran attribute access and method calls, so an expression copied from a data file
could read files or overwrite the cached frame every conversation shares (2026-10-01 audit, DF-2).
"""

from __future__ import annotations

import pandas as pd
import pytest

import tools.dataframe_cache as cache
from tools.hygiene import query_data
from tools.query_expr import ExpressionError, filter_rows

DF = pd.DataFrame({
    "a": [1.0, 2.0, 3.0, None, 5.0],
    "b": [10, 20, 30, 40, 50],
    "lith": ["granite", "basalt", "Granite gneiss", None, "schist"],
    "Au ppm": [0.1, 0.5, 1.5, 0.0, 2.0],
})


@pytest.mark.parametrize("expression, pandas_expression", [
    ("a > 1", None),
    ("a > 1 and b < 50", None),
    ("a > 1 & (b < 50)", "a > 1 & (b < 50)"),
    ("not (a > 1)", "~(a > 1)"),
    ("~(b >= 30)", None),
    ("1 < a < 5", None),
    ("b == 20 or b == 40", None),
    ("`Au ppm` >= 0.5", None),
    ("lith in ['granite', 'schist']", None),
    ("lith not in ['granite']", None),
    ("b * 2 > 50", None),
    ("b / 10 + 1 >= 4", None),
    ("b % 20 == 0", None),
    ("-a < -2", None),
    ("a > b / 10", None),
    ("lith == 'basalt'", None),
    ("lith == ['granite', 'schist']", None),
    ("lith != ['granite']", None),
    ("['granite', 'schist'] == lith", None),
    ("b == [10, 50]", None),
])
def test_matches_pandas_query(expression, pandas_expression):
    assert filter_rows(DF, expression).equals(DF.query(pandas_expression or expression))


@pytest.mark.parametrize("expression, rows", [
    ("lith.str.contains('gran', case=False)", [0, 2]),
    ("lith.str.startswith('gran')", [0]),
    ("a.notna()", [0, 1, 2, 4]),
    ("a.isna()", [3]),
    ("b.between(20, 40)", [1, 2, 3]),
    ("b.isin([10, 50])", [0, 4]),
    ("lith == \"x`y\"", []),
    ("lith.str.contains('gran|schist')", [0, 4]),
    ("lith.str.contains('^gran|ist$')", [0, 4]),
    ("lith.str.contains('GRAN|BASALT', case=False)", [0, 1, 2]),
    ("lith.str.contains('^granite$')", [0]),
    ("lith.str.contains('a.b', regex=False)", []),
    ("lith.str.contains('gneiss\\\\.?', regex=True)", None),
])
def test_column_methods(expression, rows):
    if rows is None:  # not a literal pattern: refused
        with pytest.raises(ExpressionError, match="str.contains takes text"):
            filter_rows(DF, expression)
        return
    assert filter_rows(DF, expression).index.tolist() == rows


@pytest.mark.parametrize("payload", [
    "@df.drop(columns=['b'], inplace=True)",
    "a.values.base.fill(0)",
    "a.array._ndarray.fill(0)",
    "a._mgr.blocks[0].values.fill(0)",
    "b.to_numpy().base.fill(0)",
    "__import__('os').system('true')",
    "@pd.read_csv('/etc/hostname')",
    "a.__class__ == 1",
    "a[0] > 1",
    "(lambda: 1)() == 1",
    "9 ** 9 ** 9 > a",
    "'x' * 1000000000 == lith",
    "lith.str.lower() == 'granite'",
    "a.apply(print) > 0",
    "b.isin(b)",
    "a.between(1, b)",
    "lith.str.contains('gran', flags=__import__('re').I)",
    "lith.str.contains('gran', flags=128)",
    "a is None",
    "[x for x in b]",
])
def test_anything_else_is_refused(payload):
    before = DF.copy(deep=True)
    with pytest.raises(ExpressionError):
        filter_rows(DF, payload)
    assert DF.equals(before)


@pytest.mark.parametrize("expression", ["a * 2", "b", "1 == 1", "", "nope > 1", "`Au ppm > 1"])
def test_non_conditions_and_unknown_columns_are_errors(expression):
    with pytest.raises(ExpressionError):
        filter_rows(DF, expression)


def test_tool_reports_refusals_and_leaves_the_shared_cache_intact(workspace):
    cache.drop()
    path = workspace / "assay.csv"
    DF.to_csv(path, index=False)
    out = query_data(str(path), "filter_summary", expression="a.values.base.fill(0)")
    assert isinstance(out, str) and out.startswith("Error in query_data filter_summary:")
    assert cache.load(str(path))["a"].tolist()[:3] == [1.0, 2.0, 3.0]
    ok = query_data(str(path), "filter_summary", expression="`Au ppm` > 1")
    assert ok["matching_rows"] == 2 and ok["total_rows"] == 5
    cache.drop()


def test_list_equality_is_membership_even_when_lengths_match():
    two = DF.head(2)
    assert filter_rows(two, "lith == ['basalt', 'granite']").equals(two.query("lith == ['basalt', 'granite']"))
    assert filter_rows(two, "lith != ['basalt', 'granite']").empty


@pytest.mark.parametrize("pattern", ["(.|.)*Z", ".*.*.*.*.*.*Z", "a+", "[gG]ran", "gran{2}", "(gran)", "gran\\d"])
def test_regexes_that_could_backtrack_are_refused_without_running(pattern):
    long = pd.DataFrame({"lith": ["x" * 40]})
    with pytest.raises(ExpressionError, match="str.contains takes text|escaped punctuation"):
        filter_rows(long, f"lith.str.contains({pattern!r})")


@pytest.mark.parametrize("payload", [
    "b > 1 and 'x' * 1500000000 == 'y'",
    "b > 1 or [0] * 150000000 == [1]",
    "b > 1 and '%1500000000d' % 1 == 'y'",
    "lith * 3 == 'xxx'",
    "lith + lith == 'x'",
    "lith * b == 'x'",
    "(b > 1) * 2 > 0",
])
def test_arithmetic_only_on_numbers_and_numeric_columns(payload):
    with pytest.raises(ExpressionError, match="numeric columns and numbers only"):
        filter_rows(DF, payload)
