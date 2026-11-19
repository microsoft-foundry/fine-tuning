import ast
import json
import math
import operator
from collections import Counter

_MAX_EXPRESSION_CHARS = 4096
_MAX_AST_NODES = 256
_MAX_AST_DEPTH = 64

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _parse_expression(expression):
    if not isinstance(expression, str) or not 0 < len(expression) <= _MAX_EXPRESSION_CHARS:
        raise ValueError("Expression length is outside the supported limit")
    tree = ast.parse(expression, mode="eval")
    pending = [(tree, 1)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
            raise ValueError("Expression AST is outside the supported complexity limit")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _evaluate(node, numbers):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, numbers)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("Numeric literals must be finite")
        numbers.append(value)
        return value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        value = _OPERATORS[type(node.op)](
            _evaluate(node.left, numbers),
            _evaluate(node.right, numbers),
        )
        if not math.isfinite(value):
            raise ValueError("Calculated values must be finite")
        return value
    raise ValueError("Only numeric literals and +, -, *, / are allowed")


def grade(sample, item) -> float:
    try:
        output = sample.get("output_json")
        if output is None:
            output = json.loads(sample["output_text"])
        expression = output["expression"]
        used_numbers = []
        calculated = _evaluate(_parse_expression(expression), used_numbers)
        input_numbers = [float(value) for value in json.loads(item["nums"])]
        reported = float(output["result"])
        target = float(item["target"])
        if not all(math.isfinite(value) for value in (
            calculated, reported, target, *used_numbers, *input_numbers,
        )):
            return 0.0
        if Counter(used_numbers) != Counter(input_numbers):
            return 0.0
        result_difference = abs(calculated - reported)
        difference = abs(calculated - target)
        if not math.isfinite(result_difference) or not math.isfinite(difference):
            return 0.0
        if result_difference > 1e-9:
            return 1.0
        if difference < 1e-9:
            return 5.0
        if difference <= 1:
            return 4.0
        if difference <= 5:
            return 3.0
        return 2.0
    except (
        KeyError,
        TypeError,
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
        RecursionError,
        json.JSONDecodeError,
    ):
        return 0.0

