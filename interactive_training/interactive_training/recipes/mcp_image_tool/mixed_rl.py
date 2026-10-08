"""Bounded, signal-gated RL for mixed-color verification tasks."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict

import chz

from interactive_training.recipes.mcp_image_tool.data import build_mixed_tasks
from interactive_training.rl.metric_util import RLTestSetEvaluator
from interactive_training.rl.train import (
    _derive_group_seed,
    do_group_rollout_and_filter_constant_reward,
    do_train_step_and_get_sampling_client,
    gather_with_progress,
)


def record(log_path, filename, row):
    with open(os.path.join(log_path, filename), "a") as output:
        output.write(json.dumps(row) + "\n")


async def rollout_with_budget(operation, remaining_seconds):
    timer = asyncio.timeout(remaining_seconds)
    try:
        async with timer:
            return await operation
    except TimeoutError:
        if not timer.expired():
            raise
        return None


async def evaluate_repeated(sampler, dataset, *, max_tokens, repetitions,
                            expected_episodes, log_path, stage, step, seed):
    results = []
    for repetition in range(repetitions):
        metrics = await RLTestSetEvaluator(
            dataset, max_tokens=max_tokens, num_groups_to_log=0,
            temperature=0.0, seed=seed + repetition,
        )(sampler)
        record(log_path, "evaluation_passes.jsonl", {
            "stage": stage, "step": step, "repetition": repetition, **metrics,
        })
        if metrics.get("test/env/all/total_episodes") != expected_episodes:
            raise RuntimeError(f"Incomplete {stage} evaluation")
        results.append(metrics)
    averaged = {
        key: sum(row[key] for row in results) / repetitions
        for key in results[0]
        if key.startswith("test/env/")
        and all(isinstance(row.get(key), (int, float)) for row in results)
    }
    summary = {"stage": stage, "step": step, "repetitions": repetitions, **averaged}
    record(log_path, "evaluations.jsonl", summary)
    return summary


async def train_mixed(config, training_client, tokenizer, train_dataset, test_dataset,
                      *, task_count, repetitions, expected_episodes):
    tasks = build_mixed_tasks(task_count, seed=config.sampling_seed or 0)
    builders = [chz.replace(train_dataset.builders[0], task=task) for task in tasks]
    batch_size = train_dataset.batch_size
    max_batches = (len(builders) + batch_size - 1) // batch_size
    if config.max_steps is not None:
        max_batches = min(max_batches, config.max_steps)
    record(config.log_path, "protocol.jsonl", {
        "training_tasks": [asdict(task) for task in tasks],
        "test_task_ids": [builder.task.task_id for builder in test_dataset.builders],
        "max_updates": max_batches, "group_size": builders[0].group_size,
        "groups_per_batch": batch_size, "learning_rate": config.learning_rate,
        "temperature": config.temperature, "seed": config.sampling_seed,
        "evaluation_repetitions": repetitions,
        "stop_rule": "skip zero-signal batches; stop at the batch or wall-clock budget",
    })
    sampler = await training_client.save_weights_and_get_sampling_client_async("before_rl")
    evaluation_kwargs = dict(
        max_tokens=config.max_tokens, repetitions=repetitions,
        expected_episodes=expected_episodes, log_path=config.log_path,
        seed=config.sampling_seed or 0,
    )
    before = await evaluate_repeated(
        sampler, test_dataset, stage="before_rl", step=0, **evaluation_kwargs,
    )
    started = time.monotonic()
    updates = 0
    skipped_batches = 0
    stop_reason = "budget_exhausted"
    for batch_index in range(max_batches):
        if (config.max_wall_clock_seconds is not None
                and time.monotonic() - started >= config.max_wall_clock_seconds):
            stop_reason = "wall_clock_budget"
            break
        batch = builders[batch_index * batch_size:(batch_index + 1) * batch_size]
        remaining = (None if config.max_wall_clock_seconds is None else
                 max(0.0, config.max_wall_clock_seconds - (time.monotonic() - started)))
        groups = await rollout_with_budget(gather_with_progress((
            do_group_rollout_and_filter_constant_reward(
                sampler, builder, max_tokens=config.max_tokens,
                temperature=config.temperature, do_remove_constant_reward_groups=False,
                enable_logging=False,
                seed=_derive_group_seed(config.sampling_seed, batch_index, group_index),
            ) for group_index, builder in enumerate(batch)
        ), desc=f"Mixed training probe {batch_index}",
            max_concurrency=config.max_concurrent_groups), remaining)
        if groups is None:
            stop_reason = "wall_clock_budget"
            break
        if any(group is None or len(group.trajectories_G) != builder.group_size
               for builder, group in zip(batch, groups, strict=True)):
            raise RuntimeError("Incomplete mixed training batch")
        rewards = [group.get_total_rewards() for group in groups]
        mixed = sum(max(values) > min(values) for values in rewards)
        for builder, group, values in zip(batch, groups, rewards, strict=True):
            for rollout_index, (trajectory, reward) in enumerate(zip(
                group.trajectories_G, values, strict=True,
            )):
                metrics = {}
                diagnostics = {}
                for transition in trajectory.transitions:
                    metrics.update(transition.metrics)
                    if "mcp_diagnostics" in transition.logs:
                        diagnostics = json.loads(transition.logs["mcp_diagnostics"])
                record(config.log_path, "rollouts.jsonl", {
                    "batch": batch_index, "task_id": builder.task.task_id,
                    "colors": builder.task.colors, "rollout_index": rollout_index,
                    "reward": reward, "metrics": metrics, **diagnostics,
                })
        record(config.log_path, "probe.jsonl", {
            "batch": batch_index, "mixed_groups": mixed, "total_groups": len(groups),
            "tasks": [{"task_id": builder.task.task_id, "colors": builder.task.colors,
                       "rewards": values, "trajectory_metrics": group.metrics_G}
                      for builder, values, group in zip(batch, rewards, groups, strict=True)],
        })
        if not mixed:
            skipped_batches += 1
            continue
        sampler, metrics = await do_train_step_and_get_sampling_client(
            config, batch_index, training_client, None, tokenizer, batch, groups,
        )
        updates += 1
        record(config.log_path, "updates.jsonl", {"updates": updates, **metrics})
        checkpoint_name = f"rl_step_{updates}"
        checkpoint = await training_client.save_state_async(checkpoint_name)
        await checkpoint.result_async()
        record(config.log_path, "rl_checkpoints.jsonl", {
            "updates": updates, "name": checkpoint_name,
        })
    checkpoint = await training_client.save_state_async("final")
    await checkpoint.result_async()
    after = await evaluate_repeated(
        sampler, test_dataset, stage="after_rl", step=updates, **evaluation_kwargs,
    ) if updates else dict(before, stage="after_rl")
    if not updates:
        record(config.log_path, "evaluations.jsonl", after)
    result = {
        "rl_updates": updates, "rl_stop_reason": stop_reason,
        "rl_skipped_batches": skipped_batches,
        "post_sft_accuracy": before["test/env/all/correct"],
        "rl_accuracy_gain": after["test/env/all/correct"] - before["test/env/all/correct"],
        "evaluation_repetitions": repetitions,
        "final_evaluation_reused": updates == 0,
    }
    for metric in ("tool_calls", "verified_answer", "reward/total"):
        key = f"test/env/all/{metric}"
        if key in before and key in after:
            name = metric.replace("/", "_")
            result[f"before_rl_{name}"] = before[key]
            result[f"after_rl_{name}"] = after[key]
            result[f"rl_{name}_change"] = after[key] - before[key]
    result["rl_efficiency_improved"] = bool(
        updates > 0
        and "before_rl_tool_calls" in result
        and "before_rl_verified_answer" in result
        and result["after_rl_tool_calls"] < result["before_rl_tool_calls"]
        and result["after_rl_verified_answer"] >= result["before_rl_verified_answer"]
    )
    record(config.log_path, "rl_report.jsonl", result)
    return result