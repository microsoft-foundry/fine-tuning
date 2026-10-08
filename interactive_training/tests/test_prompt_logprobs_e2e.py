"""Manual, paid-service diagnostic for prompt_logprobs support (not run by pytest).

Exercises the full flow:
  1. Create a LoRA training session
  2. Sample with prompt_logprobs enabled
  3. Forward-backward pass
  4. Optim step
  5. Save weights for sampler
  6. Compute logprobs (via sample with max_tokens=1; prompt_logprobs come from
     prefill, so the single generated token is discarded)
  7. Verify compute_post_kl from metrics.py works
  8. Close session

Usage:
    # Set the endpoint
    export AZURE_AI_PROJECT_ENDPOINT="https://<your-endpoint>"
    # Auth via API key or DefaultAzureCredential
    export AZURE_AI_API_KEY="<key>"  # optional, falls back to DefaultAzureCredential

    python tests/test_prompt_logprobs_e2e.py --model Qwen/Qwen3.8-27B
"""

import argparse
import asyncio
import logging
import math
import os
import sys

from interactive_training import model_info, tokenizer_utils

logger = logging.getLogger(__name__)


def prompt_tokens_for_model(args):
    if args.prompt_tokens is not None:
        return list(args.prompt_tokens)
    tokenizer = tokenizer_utils.get_tokenizer(args.model)
    return tokenizer.encode(args.prompt, add_special_tokens=True)


async def main(args):
    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import (
        AdamParams,
        Datum,
        LoRAConfig,
        ModelInput,
        ModelInputChunk,
        SamplingParams,
        TensorData,
    )

    endpoint = args.endpoint or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        raise ValueError("Provide --endpoint or set AZURE_AI_PROJECT_ENDPOINT")
    prompt_tokens = prompt_tokens_for_model(args)
    if len(prompt_tokens) < 2:
        raise ValueError("The prompt must contain at least two tokens")

    # --- Auth ---
    api_key = os.environ.get("AZURE_AI_API_KEY")
    if api_key:
        from azure.core.credentials import AzureKeyCredential
        credential = AzureKeyCredential(api_key)
        logger.info("Using AzureKeyCredential")
    else:
        from azure.identity.aio import DefaultAzureCredential
        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
        logger.info("Using DefaultAzureCredential")

    client_kwargs = dict(endpoint=endpoint, credential=credential)
    if not api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]

    client = FineTuningSessionClient(**client_kwargs)

    session_id = None
    try:
        # --- Step 1: Create session ---
        logger.info("Creating training session (model=%s, lora_rank=%d)...", args.model, args.lora_rank)
        session_id = await client.create_session(
            base_model=args.model,
            lora_config=LoRAConfig(rank=args.lora_rank),
            type="training",
            timeout_sec=600.0,
        )
        logger.info("Session ready: %s", session_id)

        # --- Step 2: Save initial sampler weights ---
        logger.info("Saving initial sampler weights...")
        save_task = await client.save_weights_for_sampler_async(session_id, "step0")
        save_result = await save_task
        checkpoint_id = save_result.checkpoint_id
        logger.info("Sampler checkpoint: %s", checkpoint_id)

        # --- Step 3: Sample with prompt_logprobs ---
        logger.info("Sampling with prompt_logprobs=True (prompt len=%d)...", len(prompt_tokens))

        sp = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, top_k=-1)
        sample_result = await client.sample(
            session_id,
            prompt_tokens,
            sp,
            checkpoint_id=checkpoint_id,
            num_samples=args.num_samples,
            prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )

        # Validate prompt_logprobs
        prompt_logprobs = sample_result.prompt_logprobs
        assert prompt_logprobs is not None, "prompt_logprobs should not be None"
        assert len(prompt_logprobs) == len(prompt_tokens), (
            f"prompt_logprobs length {len(prompt_logprobs)} != prompt length {len(prompt_tokens)}"
        )
        assert prompt_logprobs[0] is None, "Position 0 should be None (no prior context)"
        for i, lp in enumerate(prompt_logprobs[1:], start=1):
            assert lp is not None, f"Position {i} should not be None, got {lp}"
            assert lp <= 0.0, f"Log-prob at position {i} should be <= 0, got {lp}"

        logger.info("✓ prompt_logprobs validated: length=%d, [0]=None, rest are valid logprobs", len(prompt_logprobs))
        logger.info("  First few values: %s", prompt_logprobs[:5])

        # Validate sequences
        sequences = sample_result.sequences
        assert sequences and len(sequences) == args.num_samples, (
            f"Expected {args.num_samples} sequences, got {len(sequences) if sequences else 0}"
        )
        logger.info("✓ Got %d sampled sequences", len(sequences))

        # --- Step 4: Build training batch and do forward-backward ---
        # Build a simple datum from the first sampled sequence
        seq = sequences[0]
        generated_tokens = list(seq.tokens or [])
        generated_logprobs = list(seq.logprobs or [])
        if not generated_tokens or len(generated_tokens) != len(generated_logprobs):
            raise ValueError("Sampling must return aligned generated tokens and logprobs")
        full_sequence = prompt_tokens + generated_tokens

        # model_input = full_sequence[:-1] (right-shifted)
        model_input_tokens = full_sequence[:-1]
        # target_tokens = full_sequence[1:] (left-shifted)
        target_tokens = full_sequence[1:]
        n_tokens = len(target_tokens)
        sampling_logprobs = [0.0] * (len(prompt_tokens) - 1) + generated_logprobs
        # mask: 0 for prompt tokens (minus 1 for shift), 1 for generated tokens
        mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)
        # advantages: simple binary reward signal
        advantages = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

        datum = Datum(
            model_input=ModelInput(chunks=[ModelInputChunk(tokens=model_input_tokens)]),
            loss_fn_inputs={
                "target_tokens": TensorData(data=target_tokens),
                "logprobs": TensorData(data=sampling_logprobs),
                "advantages": TensorData(data=advantages),
            },
        )

        logger.info("Forward-backward with batch of 1 datum (seq_len=%d)...", n_tokens)
        fb_result = await client.forward_backward(
            session_id,
            [datum],
            loss_fn="importance_sampling",
        )
        logger.info("✓ Forward-backward complete")

        # --- Step 5: Optim step ---
        logger.info("Optim step...")
        optim_result = await client.optim_step(
            session_id,
            AdamParams(learning_rate=1e-5),
        )
        logger.info("✓ Optim step complete")

        # --- Step 6: Save updated sampler weights ---
        logger.info("Saving post-update sampler weights...")
        save_task = await client.save_weights_for_sampler_async(session_id, "step1")
        save_result = await save_task
        post_checkpoint_id = save_result.checkpoint_id
        logger.info("Post-update checkpoint: %s", post_checkpoint_id)

        # --- Step 7: Compute logprobs on post-update model (simulates compute_post_kl) ---
        # max_tokens must be >= 1 (the server rejects 0); prompt_logprobs come
        # from prefill, so the single generated token is discarded.
        logger.info("Computing logprobs on post-update model (max_tokens=1, prompt_logprobs=True)...")
        logprobs_sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1)
        logprobs_result = await client.sample(
            session_id,
            full_sequence,  # pass the full sequence as prompt
            logprobs_sp,
            checkpoint_id=post_checkpoint_id,
            num_samples=1,
            prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )

        new_logprobs = logprobs_result.prompt_logprobs
        assert new_logprobs is not None, "Post-update prompt_logprobs should not be None"
        assert len(new_logprobs) == len(full_sequence), (
            f"Post-update logprobs length {len(new_logprobs)} != full_sequence length {len(full_sequence)}"
        )
        assert new_logprobs[0] is None, "Position 0 should be None"

        logger.info("✓ Post-update logprobs validated: length=%d", len(new_logprobs))

        # --- Step 8: Verify KL computation (simulates compute_post_kl from metrics.py) ---
        import torch

        prev_logprobs = torch.tensor(datum.loss_fn_inputs["logprobs"].data)
        action_mask = torch.tensor(mask) > 0
        new_logprobs_tensor = torch.tensor(new_logprobs[1:])  # skip None at pos 0

        assert len(new_logprobs_tensor) == len(prev_logprobs), (
            f"new_logprobs[1:] length {len(new_logprobs_tensor)} != prev_logprobs length {len(prev_logprobs)}"
        )

        flat_diffs = (prev_logprobs - new_logprobs_tensor)[action_mask]
        kl_post_v1 = flat_diffs.mean().item()
        kl_post_v2 = 0.5 * (flat_diffs ** 2).mean().item()

        logger.info("✓ KL metrics computed successfully:")
        logger.info("  kl_pre_post_v1 = %.6f", kl_post_v1)
        logger.info("  kl_pre_post_v2 = %.6f", kl_post_v2)

        # --- Step 9: Also test without prompt_logprobs (backward compat) ---
        logger.info("Sampling WITHOUT prompt_logprobs (sanity check)...")
        no_lp_result = await client.sample(
            session_id,
            prompt_tokens,
            sp,
            checkpoint_id=post_checkpoint_id,
            num_samples=1,
            prompt_logprobs=False,
        )
        assert no_lp_result.prompt_logprobs is None, (
            f"prompt_logprobs should be None when not requested, got {no_lp_result.prompt_logprobs}"
        )
        logger.info("✓ prompt_logprobs=False correctly returns None")

        # --- Step 10: Sample with topk_prompt_logprobs ---
        topk = args.topk_prompt_logprobs
        logger.info("Sampling with topk_prompt_logprobs=%d ...", topk)
        topk_sp = SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, top_k=-1)
        topk_result = await client.sample(
            session_id,
            prompt_tokens,
            topk_sp,
            checkpoint_id=post_checkpoint_id,
            num_samples=1,
            prompt_logprobs=True,
            topk_prompt_logprobs=topk,
        )

        # Validate topk_prompt_logprobs
        topk_logprobs = topk_result.topk_prompt_logprobs
        assert topk_logprobs is not None, "topk_prompt_logprobs should not be None"
        assert len(topk_logprobs) == len(prompt_tokens), (
            f"topk_prompt_logprobs length {len(topk_logprobs)} != prompt length {len(prompt_tokens)}"
        )
        assert topk_logprobs[0] is None, "Position 0 should be None (no prior context)"
        for i, pos_entries in enumerate(topk_logprobs[1:], start=1):
            assert pos_entries is not None, f"Position {i} should not be None"
            assert len(pos_entries) <= topk, (
                f"Position {i}: expected at most {topk} entries, got {len(pos_entries)}"
            )
            for entry in pos_entries:
                assert len(entry) == 2, f"Each entry should be [token_id, logprob], got {entry}"
                assert isinstance(entry[0], (int, float)), f"token_id should be int, got {type(entry[0])}"
                assert isinstance(entry[1], float), f"logprob should be float, got {type(entry[1])}"
                assert entry[1] <= 0.0, f"logprob should be <= 0, got {entry[1]}"
            # Verify entries are sorted by logprob descending
            logprobs_only = [e[1] for e in pos_entries]
            assert logprobs_only == sorted(logprobs_only, reverse=True), (
                f"Position {i}: entries not sorted by logprob descending: {logprobs_only}"
            )

        # Also verify prompt_logprobs is still returned alongside topk
        assert topk_result.prompt_logprobs is not None, "prompt_logprobs should also be returned"
        logger.info("✓ topk_prompt_logprobs validated: length=%d, k=%d", len(topk_logprobs), topk)

        # --- Step 11: topk_prompt_logprobs=0 should return None ---
        logger.info("Sampling with topk_prompt_logprobs=0 (should return None)...")
        no_topk_result = await client.sample(
            session_id,
            prompt_tokens,
            sp,
            checkpoint_id=post_checkpoint_id,
            num_samples=1,
            prompt_logprobs=True,
            topk_prompt_logprobs=0,
        )
        assert no_topk_result.topk_prompt_logprobs is None, (
            f"topk_prompt_logprobs should be None when k=0, got {no_topk_result.topk_prompt_logprobs}"
        )
        logger.info("✓ topk_prompt_logprobs=0 correctly returns None")

        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED ✓")
        print("=" * 60)
        print(f"  Session: {session_id}")
        print(f"  Model: {args.model}")
        print(f"  Prompt length: {len(prompt_tokens)}")
        print(f"  Generated tokens: {len(generated_tokens)}")
        print(f"  prompt_logprobs length: {len(prompt_logprobs)}")
        print(f"  topk_prompt_logprobs length: {len(topk_logprobs)}")
        print(f"  KL(pre||post) v1: {kl_post_v1:.6f}")
        print(f"  KL(pre||post) v2: {kl_post_v2:.6f}")
        print("=" * 60)

    finally:
        if session_id:
            logger.info("Closing session %s...", session_id)
            try:
                await client.close_session(session_id)
                logger.info("Session closed.")
            except Exception as e:
                logger.warning("Failed to close session: %s", e)
        try:
            await client.close()
        finally:
            if not api_key:
                await credential.close()


def parse_args():
    parser = argparse.ArgumentParser(description="E2E test for prompt_logprobs support")
    parser.add_argument("--endpoint", type=str, default=None, help="Azure AI project endpoint URL")
    parser.add_argument("--model", type=str, default=model_info.DEFAULT_MODEL_NAME,
                        help="Base model name")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max generated tokens")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples per prompt")
    parser.add_argument("--topk-prompt-logprobs", type=int, default=5,
                        help="Number of top-k logprobs per prompt token (default: 5)")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=None,
                        help="Optional token IDs from the selected model's tokenizer")
    parser.add_argument("--prompt", default="This is a test", help="Text encoded by the selected model's tokenizer")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main(parse_args()))
