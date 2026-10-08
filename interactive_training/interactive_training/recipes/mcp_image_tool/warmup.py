"""Training-only supervised tool demonstrations for the alien vocabulary."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import chz
from azure.ai.finetuningsessions.models import AdamParams

from interactive_training.recipes.mcp_image_tool.data import AlienColorTask
from interactive_training.recipes.mcp_image_tool.mcp_env import AlienColorAnswerReward, MCPImageToolDataset
from interactive_training.renderers.base import Message, ToolCall
from interactive_training.supervised.common import compute_mean_nll
from interactive_training.supervised.data import conversation_to_datum
from interactive_training.tool_use.agent_tool_message_env import AgentToolMessageEnv

logger = logging.getLogger(__name__)


async def demonstrate_task(
    task: AlienColorTask, env: AgentToolMessageEnv
) -> list[Message]:
    await env.initial_observation()
    for index, word in enumerate(task.answers):
        result = await env.step(Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(
                id=f"verify_{index}",
                function=ToolCall.FunctionBody(
                    name="reveal_alien_color",
                    arguments=json.dumps({"alien_word": word}),
                ),
            )],
        ))
        if result.episode_done:
            raise ValueError("Demonstration ended before the final answer")
    answer = (
        task.answers[0] if len(task.colors) == 1 else
        ", ".join(f"{color}={word}" for color, word in zip(task.colors, task.answers, strict=True))
    )
    result = await env.step(Message(role="assistant", content=f"Answer: {answer}"))
    if not result.episode_done:
        raise ValueError("Demonstration did not finish")
    grader = AlienColorAnswerReward(
        expected=task.answers[0] if len(task.colors) == 1 else tuple(zip(task.colors, task.answers)),
        max_tool_calls=len(task.colors),
        format_coef=0.0,
    )
    reward, metrics = await grader(env.history)
    if reward != 1.0 or metrics["correct"] != 1.0:
        raise ValueError("MCP demonstration failed exact verification")
    return env.history


async def train_warmup(
    training_client,
    dataset: MCPImageToolDataset,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    log_path: str,
) -> None:
    demonstrations = []
    for builder in dataset.builders:
        env = (await chz.replace(builder, group_size=1).make_envs())[0]
        try:
            history = await demonstrate_task(builder.task, env.message_env)
            demonstrations.append(conversation_to_datum(
                history, env.renderer, max_length=builder.max_trajectory_tokens,
                fail_on_truncation=True,
            ))
        finally:
            await env.message_env.close()
    generator = random.Random(seed)
    step = 0
    with (Path(log_path) / "warmup_metrics.jsonl").open("w") as output:
        for epoch in range(epochs):
            indices = list(range(len(demonstrations)))
            generator.shuffle(indices)
            for start in range(0, len(indices), dataset.batch_size):
                batch = [demonstrations[index] for index in indices[start:start + dataset.batch_size]]
                forward = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
                result = await forward.result_async()
                optim = await training_client.optim_step_async(AdamParams(
                    learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
                ))
                await optim.result_async()
                step += 1
                metrics = {
                    "stage": "sft", "step": step, "epoch": epoch + 1,
                    "examples": len(batch),
                    "nll": compute_mean_nll(
                        [item["logprobs"] for item in result.loss_fn_outputs],
                        [datum.loss_fn_inputs["weights"] for datum in batch],
                    ),
                }
                output.write(json.dumps(metrics) + "\n")
                output.flush()
                logger.info("Supervised warm-up: %s", json.dumps(metrics))
    checkpoint = await training_client.save_state_async("warmup")
    await checkpoint.result_async()