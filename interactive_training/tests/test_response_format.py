import asyncio
from unittest.mock import AsyncMock, MagicMock

from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk
from interactive_training.completers import SessionTokenCompleter
from interactive_training.rl.metric_util import RLTestSetEvaluator


def test_session_token_completer_forwards_response_format() -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            "strict": True,
        },
    }
    result = MagicMock()
    result.sequences = [MagicMock(tokens=[1], logprobs=[-0.1])]
    sampling_client = MagicMock()
    sampling_client.sample_async = AsyncMock(return_value=result)
    completer = SessionTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=16,
        response_format=response_format,
    )

    asyncio.run(
        completer(
            ModelInput(chunks=[ModelInputChunk(tokens=[100, 200])]),
            stop=[50256],
        )
    )

    sampling_params = sampling_client.sample_async.call_args.kwargs[
        "sampling_params"
    ]
    assert sampling_params.response_format == response_format


def test_validation_evaluator_forwards_sampling_configuration() -> None:
    response_format = {"type": "json_object"}
    dataset = MagicMock()
    dataset.__len__.return_value = 0
    evaluator = RLTestSetEvaluator(
        dataset,
        max_tokens=16,
        response_format=response_format,
        temperature=0.0,
        seed=17,
    )
    evaluator.eval_token_completer = AsyncMock(return_value={})

    asyncio.run(evaluator(MagicMock(), step=1))

    policy = evaluator.eval_token_completer.call_args.args[0]
    assert policy.response_format == response_format
    assert policy.temperature == 0.0
    assert policy.seed == 17
