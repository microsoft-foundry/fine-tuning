import re


def normalize(answer):
    if not isinstance(answer, str):
        return ""
    compact = re.sub(r"\s+", "", answer.strip())
    compact = compact.replace(r"\,", ",")
    # Reorder only whole answers, never coordinates or nested LaTeX arguments.
    parts = []
    stack = []
    matching = {")": "(", "]": "[", "}": "{"}
    start = 0
    for position, character in enumerate(compact):
        if character in "([{":
            stack.append(character)
        elif character in matching:
            if not stack or stack.pop() != matching[character]:
                return compact
        elif character == "," and not stack:
            parts.append(compact[start:position])
            start = position + 1
    if stack:
        return compact
    parts.append(compact[start:])
    return ",".join(sorted(part for part in parts if part))


def extract_model_answer(text):
    if not text or not isinstance(text, str):
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth = 1
    position = start
    while position < len(text) and depth:
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
        position += 1
    return text[start : position - 1].strip() if depth == 0 else ""


def grade(sample, item) -> float:
    output = (
        sample.get("output_text", "") or sample.get("output_json", "")
        if isinstance(sample, dict)
        else getattr(sample, "output_text", "") or getattr(sample, "output_json", "")
    )
    reference = (
        item.get("answer", "") if isinstance(item, dict) else getattr(item, "answer", "")
    )
    predicted = extract_model_answer(str(output))
    if not predicted or not reference:
        return 0.0
    return 1.0 if normalize(predicted) == normalize(str(reference)) else 0.0
