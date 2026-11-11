import statistics


def compute_rollout_metrics(groups, max_tokens, prefix="env/all"):
    """Summarize successful samples; failed requests are counted by the caller.

    ``reward/total`` is this notebook's soft tool-call score, not exact-match
    accuracy; ``format`` is renderer parse success. Reward equals correctness
    here (the full tool_rl recipe also applies a format penalty).
    ``total_episodes`` counts successful samples, each of which is one turn.
    Mixed groups have unequal rewards, matching rl.metric_util. Report this
    only for grouped sampling; singleton evaluation has no group diversity.
    Detailed prompt/token accounting and redundant turn counts are omitted.
    """
    groups = [group for group in groups if group]
    samples = [sample for group in groups for sample in group]
    if not samples:
        return {}

    action_lengths = [len(sample["tokens"]) for sample in samples]
    reward = statistics.mean(sample["score"] for sample in samples)
    metrics = {
        "reward/total": reward,
        "format": statistics.mean(float(sample["parse_ok"]) for sample in samples),
        "n_pred": statistics.mean(len(sample["actual"]) for sample in samples),
        "n_gold": statistics.mean(len(sample["expected"]) for sample in samples),
        "ac_tokens_per_turn": statistics.mean(action_lengths),
        "total_episodes": len(samples),
    }
    if any(len(group) > 1 for group in groups):
        mixed = sum(len({sample["score"] for sample in group}) > 1 for group in groups)
        metrics["by_group/frac_mixed"] = mixed / len(groups)
    if max_tokens > 0:
        metrics["ac_tokens_frac_at_max"] = statistics.mean(
            float(length >= max(1, max_tokens - 4)) for length in action_lengths
        )
    return {f"{prefix}/{key}": value for key, value in metrics.items()}