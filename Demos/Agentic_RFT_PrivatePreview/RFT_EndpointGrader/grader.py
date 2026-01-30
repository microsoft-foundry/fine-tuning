"""
Python Grader logic for OpenAI Evals

Example from https://github.com/azure-ai-foundry/fine-tuning/blob/main/Demos/RFT_Countdown/demo_with_python_grader.ipynb
"""
from typing import Any, Dict
import ast
import json
import logging
import re


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


def grade(sample: Dict[str, Any], item: Dict[str, Any]) -> float:
    """
    Implements the OpenAI Python Grader API, grading the given sample in the
    context of the provided item.

    Returns a score in all cases, with 0 returned in case of error.
    """
    # Be flexible about where our output comes from, but it *must* be JSON.
    output = {}
    if "output_json" in sample:
        output = sample["output_json"]
    else:
        # fallback to output_text but it better be JSON!
        try:
            output = json.loads(sample["output_text"])
        except Exception as e:
            logging.error(f"failed to find JSON output in output_json and output_text: {e}")
            return 0
    if not output:
        logging.error("failed to find non-null JSON output in sample")
        return 0
    
    # Extract our Expression and Result
    expr = output.get("expression")
    if not expr:
        logging.warning("expression was empty")
        return 0
    result = output.get("result")
    if not result:
        logging.warning("result was empty")
        return 0

    # Perform the evaluation.
    try:
        expr_val = _safe_eval(expr)
    except ValueError as e:
        logging.error(f"value error evaluating expression: {e}")
        return 0
    
    # TODO: handle parsing item more cleanly.
    try:
        # Check if all numbers were used.
        used = sorted(map(int, re.findall(r"-?\d+", expr)))
        expected = sorted(map(int, item["nums"]))
        if used != expected:
            logging.info("all numbers were not used exactly once")
            return 0

        sr = int(float(result))
        it = int(float(item["target"]))

        # Score function
        if expr_val != sr:
            return 1
        if sr == it:
            return 5
        if abs(sr - it) <= 1:
            return 4
        if abs(sr - it) <= 5:
            return 3
        return 2
    except Exception as e:
        logging.error(f"exception while grading: {e}")
    return 0


if __name__ == "__main__":
    import json
    data = {}
    try:
        with open("./test.json", mode="rb") as f:
            data = json.load(f)
        sample, item = data["sample"], data["item"]
        
        print(f"grading with:\n\tsample: {sample}\n\titem: {item}")
        score = grade(sample, item)
        print(f"score: {score}")
    except Exception as e:
        import sys
        print(e, file=sys.stderr)
