from __future__ import annotations

import ast
from typing import Any


class ConstraintError(ValueError):
    pass


def evaluate_constraint(expr: str, params: dict[str, Any]) -> bool:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # noqa: BLE001
        raise ConstraintError(f"Invalid constraint syntax: {expr}") from exc
    return bool(_eval_node(node.body, params))


def _eval_node(node: ast.AST, params: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ConstraintError(f"Unknown parameter in constraint: {node.id}")
        return params[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        val = _eval_node(node.operand, params)
        if isinstance(node.op, ast.Not):
            return not bool(val)
        if isinstance(node.op, ast.USub):
            return -val
        return +val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_node(node.left, params)
        right = _eval_node(node.right, params)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_eval_node(v, params) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(v) for v in values)
        return any(bool(v) for v in values)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, params)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval_node(comparator, params)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            else:  # pragma: no cover - unsupported op
                raise ConstraintError("Unsupported comparison operator")
            if not ok:
                return False
            left = right
        return True
    raise ConstraintError("Unsupported constraint expression")

