"""Row filters for query_data(operation="filter_summary"), evaluated without df.query.

df.query runs here on pandas' python engine (numexpr is not installed), which allows attribute access and method
calls. Text copied from a data file could then read files outside the workspace ('@pd.read_csv("/etc/...")') or
overwrite the cached DataFrame every conversation shares ('Au_ppm.values.base.fill(0)', no '@' needed). Checking the
string and then handing it to df.query is not enough: the check and pandas would have to tokenise every edge case
(a backtick inside a string literal) the same way. So the expression is parsed with ast and evaluated here, and only
these nodes exist:

- column names, bare or `quoted with backticks`;
- numbers, strings, True/False/None, and lists or tuples of them;
- comparisons ==, !=, <, <=, >, >=, in, not in (chains allowed);
- and, or, not, &, |, ~ (& and | bind like and/or, as in pandas' query);
- + - * / // % ** and unary minus, on numeric columns and numbers only (string or list repetition could allocate
  gigabytes in the process every conversation shares);
- the column methods isna, isnull, notna, notnull, between, isin, and str.contains, str.startswith, str.endswith,
  all with constant arguments;
- col == [..] and col != [..] mean isin / not isin, as in pandas' query.

str.contains takes a literal, or literals joined by | and optionally anchored with ^ or $ ('gran|basalt',
'^DH0'). Any other regex is refused: Python's re holds the GIL, so a backtracking pattern ('(.|.)*Z') would freeze
the whole shared server, not only the conversation that sent it.
"""

from __future__ import annotations

import ast
import operator

MAX_LENGTH = 2000
_POW_LIMIT = 64  # exponent bound when both operands are plain numbers (9**9**9 would never finish)

_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_ARITHMETIC = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_METHODS = {"isna", "isnull", "notna", "notnull", "between", "isin"}
_STR_METHODS = {"contains", "startswith", "endswith"}
_KEYWORDS = {"case", "na", "regex", "inclusive"}
_REGEX_META = set(".^$*+?{}[]\\|()")
ALLOWED = (
    "column comparisons (==, !=, <, <=, >, >=, in, not in), and/or/not, &, |, ~, arithmetic, and the methods "
    "isna/notna/between/isin/str.contains/str.startswith/str.endswith with constant arguments"
)


class ExpressionError(ValueError):
    pass


def _literal_alternatives(pattern: str):
    """
    'gran|^DH0|ite$' -> [(text, anchored_start, anchored_end), ...]. Backslash-escaped metacharacters are literal.
    Raises ExpressionError for any other regex syntax.
    """
    alternatives, current, i = [], [], 0
    start = end = False
    while i <= len(pattern):
        c = pattern[i] if i < len(pattern) else "|"
        if c == "|":
            if not current and not (start or end):
                raise ExpressionError(f"empty alternative in {pattern!r}")
            alternatives.append(("".join(current), start, end))
            current, start, end = [], False, False
        elif c == "\\":
            if i + 1 >= len(pattern) or pattern[i + 1] not in _REGEX_META:
                raise ExpressionError(f"str.contains: only escaped punctuation like \\. is allowed in {pattern!r}")
            current.append(pattern[i + 1])
            i += 1
        elif c == "^" and not current and not start:
            start = True
        elif c == "$" and (i + 1 == len(pattern) or pattern[i + 1] == "|"):
            end = True
        elif c in _REGEX_META:
            raise ExpressionError(
                f"str.contains takes text, or texts joined by | (optionally ^ or $ anchored), not the regex {pattern!r}"
            )
        else:
            current.append(c)
        i += 1
    return alternatives


def _contains(series, pattern, case=True, na=None, regex=True):
    """str.contains without the regex engine: a literal, or literal alternatives with optional ^/$ anchors."""
    if not isinstance(pattern, str) or not isinstance(case, bool) or not isinstance(regex, bool):
        raise ExpressionError("str.contains needs a text pattern, and case/regex must be True or False")
    text = series.str
    if not case:
        text = series.str.lower().str
        pattern = pattern.lower()
    alternatives = _literal_alternatives(pattern) if regex else [(pattern, False, False)]
    result = None
    for literal, start, end in alternatives:
        if start and end:
            part = (series.str.lower() if not case else series) == literal
        elif start:
            part = text.startswith(literal)
        elif end:
            part = text.endswith(literal)
        else:
            part = text.contains(literal, regex=False)
        part = part.astype("boolean").fillna(False if na is None else bool(na)).astype(bool)
        result = part if result is None else result | part
    return result


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _prepare(expression: str):
    """
    Outside string literals: `quoted names` -> placeholder identifiers, and & | -> and / or, as pandas' query parser
    does (so 'a > 1 & b < 5' means (a > 1) and (b < 5), not Python's a > (1 & b) < 5).
    Returns (text, {placeholder: name}).
    """
    out, names, i, quote = [], {}, 0, None
    while i < len(expression):
        c = expression[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(expression):
                out.append(expression[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            out.append(c)
        elif c == "`":
            end = expression.find("`", i + 1)
            if end < 0:
                raise ExpressionError("a backtick-quoted column name is not closed")
            placeholder = f"_bt{len(names)}_"
            names[placeholder] = expression[i + 1:end]
            out.append(placeholder)
            i = end + 1
            continue
        elif c in "&|":
            out.append(" and " if c == "&" else " or ")
        else:
            out.append(c)
        i += 1
    return "".join(out), names


class _Evaluator:
    def __init__(self, df, quoted: dict):
        self.df = df
        self.quoted = quoted

    def visit(self, node):
        handler = getattr(self, f"_{type(node).__name__}", None)
        if handler is None:
            raise ExpressionError(f"'{ast.unparse(node)}' is not allowed; use {ALLOWED}")
        return handler(node)

    def _Expression(self, node):
        return self.visit(node.body)

    def _Name(self, node):
        name = self.quoted.get(node.id, node.id)
        if name in self.df.columns:
            return self.df[name]
        shown = ", ".join(map(str, list(self.df.columns)[:40]))
        raise ExpressionError(f"unknown column {name!r} (put names with spaces in backticks). Columns: {shown}")

    def _Constant(self, node):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise ExpressionError(f"constant {node.value!r} is not allowed")

    def _constant(self, node):
        if isinstance(node, ast.Constant):
            return self._Constant(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
            return -node.operand.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._constant(e) for e in node.elts]
        raise ExpressionError(f"'{ast.unparse(node)}' must be a constant (a number, a string, or a list of them)")

    def _List(self, node):
        return self._constant(node)

    _Tuple = _List

    def _UnaryOp(self, node):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return (not value) if isinstance(value, bool) else ~value
        raise ExpressionError(f"'{ast.unparse(node)}' is not allowed; use {ALLOWED}")

    def _BinOp(self, node):
        import pandas as pd

        op = _ARITHMETIC.get(type(node.op))
        if op is None:
            raise ExpressionError(f"'{ast.unparse(node)}' is not allowed; use {ALLOWED}")
        left, right = self.visit(node.left), self.visit(node.right)
        for value in (left, right):  # numeric columns and numbers only: str/list repetition is unbounded memory
            numeric_column = isinstance(value, pd.Series) and pd.api.types.is_numeric_dtype(value) \
                and not pd.api.types.is_bool_dtype(value)
            if not (numeric_column or _is_number(value)):
                raise ExpressionError(f"'{ast.unparse(node)}': arithmetic works on numeric columns and numbers only")
        if _is_number(left) and _is_number(right) and isinstance(node.op, ast.Pow) \
                and (abs(right) > _POW_LIMIT or abs(left) > 1e6):
            raise ExpressionError(f"'{ast.unparse(node)}': exponent too large")
        return op(left, right)

    def _BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        result = values[0]
        for value in values[1:]:
            result = combine(result, value)
        return result

    def _Compare(self, node):
        left, result = self.visit(node.left), None
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(right, list) or not hasattr(left, "isin"):
                    raise ExpressionError("'in' needs a column and a list, e.g. lithology in ['granite', 'basalt']")
                part = left.isin(right)
                if isinstance(op, ast.NotIn):
                    part = ~part
            elif isinstance(op, (ast.Eq, ast.NotEq)) and (isinstance(left, list) or isinstance(right, list)):
                # pandas' query reads col == [..] as isin (an element-wise compare would be silently wrong)
                column, values = (right, left) if isinstance(left, list) else (left, right)
                if not hasattr(column, "isin") or isinstance(values, type(column)):
                    raise ExpressionError("'== [..]' needs a column and a list, e.g. lithology == ['granite', 'basalt']")
                part = column.isin(values)
                if isinstance(op, ast.NotEq):
                    part = ~part
            elif type(op) in _COMPARE:
                part = _COMPARE[type(op)](left, right)
            else:
                raise ExpressionError(f"'{ast.unparse(node)}' is not allowed (use == or != instead of is)")
            result = part if result is None else result & part
            left = right
        return result

    def _Call(self, node):
        func = node.func
        if not isinstance(func, ast.Attribute):
            raise ExpressionError(f"'{ast.unparse(node)}': only column methods can be called; use {ALLOWED}")
        args = [self._constant(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg not in _KEYWORDS:
                raise ExpressionError(f"'{ast.unparse(node)}': keyword {kw.arg!r} is not allowed")
            kwargs[kw.arg] = self._constant(kw.value)
        if isinstance(func.value, ast.Attribute) and func.value.attr == "str" and func.attr in _STR_METHODS:
            series = self._column(func.value.value)
            if func.attr == "contains":
                return _contains(series, *args, **kwargs)
            if not args or not all(isinstance(a, str) for a in args[:1]) or set(kwargs) - {"na"}:
                raise ExpressionError(f"'{ast.unparse(node)}': str.{func.attr} takes one text argument")
            return getattr(series.str, func.attr)(*args, **kwargs)
        if func.attr in _METHODS:
            return getattr(self._column(func.value), func.attr)(*args, **kwargs)
        raise ExpressionError(f"'{ast.unparse(node)}' is not allowed; use {ALLOWED}")

    def _column(self, node):
        if not isinstance(node, ast.Name):
            raise ExpressionError(f"'{ast.unparse(node)}': methods apply to a column name")
        return self._Name(node)


def filter_rows(df, expression: str):
    """Rows of ``df`` matching ``expression`` (see the module docstring for the syntax)."""
    import numpy as np
    import pandas as pd

    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("the expression is empty")
    if len(expression) > MAX_LENGTH:
        raise ExpressionError(f"the expression is longer than {MAX_LENGTH} characters")
    text, quoted = _prepare(expression)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        hint = " (@variables are not supported)" if "@" in text else ""
        raise ExpressionError(f"not a valid expression: {exc.msg}{hint}") from None
    try:
        mask = _Evaluator(df, quoted).visit(tree)
    except ExpressionError:
        raise
    except Exception as exc:  # noqa: BLE001 - e.g. .str on a numeric column, comparing str to a number
        raise ExpressionError(f"could not evaluate {expression!r}: {exc}") from None
    not_condition = ExpressionError(f"the expression must be a condition on columns, e.g. Au_ppm > 0.5; got {expression!r}")
    if not isinstance(mask, pd.Series) or len(mask) != len(df):
        raise not_condition
    if not pd.api.types.is_bool_dtype(mask.dtype):
        # str methods give object dtype when the column has missing values: True/False/NaN is still a condition
        if mask.dtype != object or not mask.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise not_condition
    return df[mask.fillna(False).astype(bool)]
