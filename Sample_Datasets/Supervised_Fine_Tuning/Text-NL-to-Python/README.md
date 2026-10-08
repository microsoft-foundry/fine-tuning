# Notice

This is sample training & validation data for text-to-Python code generation. The data was synthetically generated using [NVIDIA Data Designer](https://developer.nvidia.com/data-designer) with GPT-5.4 as the generator model.

## Dataset Summary

This dataset contains (instruction, Python code) pairs for supervised fine-tuning. Each example includes a natural language programming task and a corresponding Python solution, formatted as chat messages (system, user, assistant).

- **Training set**: 85 examples
- **Validation set**: 4 examples
- **Generation method**: GPT-5.4 via NVIDIA Data Designer with automated quality judges
- **Quality filtering**: Only examples scoring ≥3.0/4.0 average across relevance, Pythonic style, readability, and efficiency judges are included
- **Syntax validation**: All code examples pass Python AST parsing

## Diversity

Examples span 3 industries (Healthcare, Finance, Technology), 3 complexity levels (Beginner, Intermediate, Advanced), and 13 coding concepts (Variables, OOP, Generators, Web Frameworks, etc.).

## Use Case

Fine-tune a smaller model (e.g., GPT-4.1-mini) to generate Python code from natural language instructions, distilling the capability of a larger teacher model (GPT-5.4).

## Format

Standard chat messages JSONL — each line is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that writes clean, well-structured Python code to solve the user's request."},
    {"role": "user", "content": "Write a function that..."},
    {"role": "assistant", "content": "def solve(...):\n    ..."}
  ]
}
```

## Related Demo

See the full end-to-end notebook in [Demos/NL_to_Python_Distillation](../../Demos/NL_to_Python_Distillation/) which walks through data generation, fine-tuning, deployment, and evaluation.

## Languages

The text and code in the dataset are in English/Python. The associated BCP-47 code is `en`.
