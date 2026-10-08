import asyncio

from rollout_metrics import compute_rollout_metrics
from sampling import sample_row


async def run_eval(sampling_client, rows, config, renderer, name="eval"):
    sem = asyncio.Semaphore(config.eval_sample_concurrency)

    async def run_one(index, row):
        async with sem:
            samples = await sample_row(
                sampling_client,
                row,
                config,
                renderer,
                temperature=config.eval_temperature,
                seed=config.seed + index,
            )
            return samples[0]

    raw_results = await asyncio.gather(
        *[run_one(index, row) for index, row in enumerate(rows)],
        return_exceptions=True,
    )
    failures = [result for result in raw_results if isinstance(result, Exception)]
    results = [result for result in raw_results if not isinstance(result, Exception)]
    if failures:
        print(f"{name}: skipped {len(failures)} failed sample requests; first error: {failures[0]}")
    if not results:
        raise RuntimeError(f"{name}: all eval sample requests failed")

    metrics = compute_rollout_metrics(
        [[result] for result in results], config.max_tokens, prefix="test/env/all"
    )
    metrics.update({
        "eval/reward": metrics["test/env/all/reward/total"],
        "eval/failed_samples": len(failures),
    })
    return metrics, sorted(results, key=lambda result: result["score"])[:5]
