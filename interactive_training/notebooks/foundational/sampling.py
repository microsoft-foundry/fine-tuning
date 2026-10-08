import asyncio

from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk, SamplingParams

from grader import grade_tool_calls
from xlam_utils import (
    renderer_tool_calls_to_grader,
    xlam_expected_tool_calls,
    xlam_tools_to_specs,
)


TOOL_CALL_SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set "
    "of possible functions. Based on the question, you will need to make one or "
    "more function/tool calls to achieve the purpose. If none of the functions "
    "can be used, point it out. If the given question lacks the parameters "
    "required by the function, also point it out."
)


def build_prompt_tokens(row, renderer):
    messages = renderer.create_conversation_prefix_with_tools(
        tools=xlam_tools_to_specs(row),
        system_prompt=TOOL_CALL_SYSTEM_PROMPT,
    )
    messages.append({"role": "user", "content": row["query"]})
    model_input = renderer.build_generation_prompt(messages)
    return [token for chunk in model_input.chunks for token in chunk.tokens]


async def sample_row(
    sampling_client,
    row,
    config,
    renderer,
    num_samples=1,
    temperature=1.0,
    seed=None,
):
    prompt_tokens = build_prompt_tokens(row, renderer)
    params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        stop_criteria=renderer.get_stop_sequences(),
        seed=seed,
    )
    try:
        result = await asyncio.wait_for(
            sampling_client.sample_async(
                ModelInput(chunks=[ModelInputChunk(tokens=prompt_tokens)]),
                num_samples=num_samples,
                sampling_params=params,
            ),
            timeout=config.sample_timeout_sec,
        )
    except Exception as exc:
        query = str(row.get("query", ""))[:200].replace("\n", " ")
        raise RuntimeError(
            f"sample failed: prompt_tokens={len(prompt_tokens)} num_samples={num_samples} "
            f"sampler={getattr(sampling_client, 'model_path', None)!r} query={query!r}"
        ) from exc

    expected = xlam_expected_tool_calls(row)
    samples = []
    for sequence in result.sequences:
        tokens = list(sequence.tokens or [])
        logprobs = list(sequence.logprobs or [])
        message, parse_ok = renderer.parse_response(tokens)
        actual = renderer_tool_calls_to_grader(list(message.get("tool_calls") or []))
        score = grade_tool_calls(actual, expected)
        samples.append(
            dict(
                prompt_tokens=prompt_tokens,
                tokens=tokens,
                logprobs=logprobs,
                score=score,
                parse_ok=parse_ok,
                actual=actual,
                expected=expected,
                query=row["query"],
            )
        )
    return samples