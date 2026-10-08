from collections.abc import Sequence

from azure.ai.finetuningsessions.models import Datum, ModelInput, ModelInputChunk, TensorData


def datums_from_groups(groups):
    datums = []
    rewards = []
    for samples in groups:
        group_rewards = [sample["score"] for sample in samples]
        group_advantages = compute_group_relative_advantages(group_rewards)
        rewards.extend(group_rewards)
        for sample, advantage in zip(samples, group_advantages, strict=True):
            datum = datum_from_sample(sample, advantage)
            if datum is not None:
                datums.append(datum)

    if not datums:
        raise RuntimeError("No datums built from sampled groups")

    return datums, rewards


def compute_group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    """Compute mean-centered, group-relative advantages for one prompt.

    For each reward ``r_i`` in a group, this returns ``r_i - mean(rewards)``.
    This is the primitive GRPO-style baseline used by the notebook: it compares
    completions sampled for the same prompt without a learned value model. It
    intentionally does not normalize by the group standard deviation or apply
    a reference-policy KL penalty.
    """
    if not rewards:
        return []

    baseline = sum(rewards) / len(rewards)
    return [float(reward - baseline) for reward in rewards]


def datum_from_sample(sample, advantage):
    action_tokens = sample["tokens"]
    action_logprobs = sample["logprobs"]
    if not action_tokens or len(action_tokens) != len(action_logprobs):
        return None

    full_tokens = sample["prompt_tokens"] + action_tokens
    if len(full_tokens) < 2:
        return None

    prompt_targets = max(len(sample["prompt_tokens"]) - 1, 0)
    logprobs = [0.0] * prompt_targets + [float(value) for value in action_logprobs]
    advantages = [0.0] * prompt_targets + [float(advantage)] * len(action_tokens)
    target_tokens = full_tokens[1:]
    input_tokens = full_tokens[:-1]

    assert len(target_tokens) == len(logprobs) == len(advantages)
    return Datum(
        model_input=ModelInput(chunks=[ModelInputChunk(tokens=input_tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens),
            "logprobs": TensorData(data=logprobs),
            "advantages": TensorData(data=advantages),
        },
    )
