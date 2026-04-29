"""Sample tool: safe arithmetic expression evaluator.

To enable, copy this file into ./free_agent_tools/. The implementation walks
the AST and only allows numeric literals + arithmetic operators — `eval()` is
NEVER called on the raw string.
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

_OPS: dict[type[ast.operator] | type[ast.unaryop], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return its numeric result as a string.

    Supports `+ - * / // % **`, parentheses, integer and float literals.
    Use this whenever the user asks for a precise computation — do not estimate
    in your head.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    try:
        result = _eval(tree.body)
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        return f"math error: {exc}"
    except _UnsafeNode as exc:
        return f"refused: only arithmetic on numbers is allowed ({exc})"
    return str(result)


class _UnsafeNode(Exception):
    pass


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise _UnsafeNode(f"non-numeric literal: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        return _OPS[type(node.op)](left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        operand = _eval(node.operand)
        return _OPS[type(node.op)](operand)  # type: ignore[operator]
    raise _UnsafeNode(node.__class__.__name__)
