# Notice

This issample training & validation data This was curated using [this](https://huggingface.co/datasets/openai/gsm8k) as the reference dataset.

## Dataset Summary

GSM8K (Grade School Math 8K) is a dataset of high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
- These problems take between 2 and 8 steps to solve.
- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues""

## Supported Tasks and Leaderboards

This dataset is generally used to test logic and math in language modelling.
It has been used for many benchmarks, including the [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Languages

The text in the dataset is in English. The associated BCP-47 code is `en`.
