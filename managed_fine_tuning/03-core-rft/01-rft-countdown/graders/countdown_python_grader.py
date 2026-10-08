import ast
import json
import operator
from collections import Counter

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate(node, numbers):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, numbers)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        numbers.append(float(node.value))
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](
            _evaluate(node.left, numbers),
            _evaluate(node.right, numbers),
        )
    raise ValueError("Only numeric literals and +, -, *, / are allowed")


def grade(sample, item) -> float:
    try:
        output = sample.get("output_json")
        if output is None:
            output = json.loads(sample["output_text"])
        expression = output["expression"]
        used_numbers = []
        calculated = _evaluate(ast.parse(expression, mode="eval"), used_numbers)
        expected_numbers = Counter(float(value) for value in json.loads(item["nums"]))
        if Counter(used_numbers) != expected_numbers:
            return 0.0
        reported = float(output["result"])
        target = float(item["target"])
        if abs(calculated - reported) > 1e-9:
            return 1.0
        difference = abs(calculated - target)
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
        json.JSONDecodeError,
    ):
        return 0.0

