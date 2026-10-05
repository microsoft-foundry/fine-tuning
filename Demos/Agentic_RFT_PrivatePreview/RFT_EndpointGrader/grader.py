"""Python grader logic for the countdown reinforcement fine-tuning demo."""

import ast
import json


def _eval(n):
    if isinstance(n, ast.Constant):
        return n.value

    if isinstance(n, ast.BinOp) and type(n.op) in {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }:
        return {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
        }[type(n.op)](_eval(n.left), _eval(n.right))

    if isinstance(n, ast.UnaryOp) and type(n.op) in {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }:
        return {
            ast.UAdd: lambda a: +a,
            ast.USub: lambda a: -a,
        }[
            type(n.op)
        ](_eval(n.operand))

    raise ValueError("bad expr")


def _safe_eval(e):
    return _eval(ast.parse(e, mode="eval").body)


def _numbers_in_expression(expression):
    numbers = []
    for node in ast.walk(ast.parse(expression, mode="eval")):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            numbers.append(abs(node.value))
    return numbers


def grade(sample, item):
    """
    Implements the OpenAI Python Grader API, grading the given sample in the
    context of the provided item.

    Returns a score in all cases, with 0 returned in case of error.
    """
    output = {}
    if "output_json" in sample:
        output = sample["output_json"]
    else:
        try:
            output = json.loads(sample["output_text"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return 0
    if not output:
        return 0

    expr = output.get("expression")
    if not isinstance(expr, str) or not expr:
        return 0
    result = output.get("result")
    if result is None:
        return 0

    try:
        expr_val = _safe_eval(expr)
        used = sorted(map(int, _numbers_in_expression(expr)))
        nums = item["nums"]
        if isinstance(nums, str):
            nums = json.loads(nums)
        expected = sorted(map(int, nums))
        if used != expected:
            return 0

        sr = int(float(result))
        it = int(float(item["target"]))

        if expr_val != sr:
            return 0.2
        if sr == it:
            return 1.0
        if abs(sr - it) <= 1:
            return 0.8
        if abs(sr - it) <= 5:
            return 0.6
        return 0.4
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        OverflowError,
    ):
        return 0
    return 0
