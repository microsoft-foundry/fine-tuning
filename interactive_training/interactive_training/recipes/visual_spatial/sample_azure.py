"""Sample one Do-You-See-Me visual-spatial example."""

from __future__ import annotations

import argparse
import os

from azure.ai.finetuningsessions import FineTuningSession, FineTuningSessionClient
from azure.ai.finetuningsessions.models import LoRAConfig, SamplingParams

from interactive_training import model_info
from interactive_training.image_processing_utils import get_image_processor
from interactive_training.recipes.visual_spatial import azure_openai_jsonl
from interactive_training.recipes.visual_spatial.data import answers_match, load_dataset, user_message
from interactive_training.renderers import get_renderer, get_text_content
from interactive_training.tokenizer_utils import get_tokenizer
from interactive_training.training_types import TRAINING_TYPES, normalize_training_type


def _credential():
    api_key = os.environ.get("AZURE_AI_API_KEY")
    if api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(api_key), {}

    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential(), {
        "credential_scopes": ["https://ai.azure.com/.default"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-endpoint", required=True)
    parser.add_argument("--model-name", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--renderer-name")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--training-type",
        type=normalize_training_type,
        choices=TRAINING_TYPES,
        help="Optional training tier; omitted values use the server default.",
    )
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--data-path")
    args = parser.parse_args()

    if args.data_path:
        _, test_dataset = azure_openai_jsonl.load_dataset(
            args.data_path,
            seed=args.seed,
        )
    else:
        _, test_dataset = load_dataset(seed=args.seed)
    row = test_dataset[args.example_index]

    model_info.require_vision_language_model(
        args.model_name,
        workload="visual-spatial sampling",
    )
    tokenizer_name = args.tokenizer_name or args.model_name
    renderer_name = (
        args.renderer_name
        or model_info.get_recommended_renderer_name(args.model_name)
    )
    tokenizer = get_tokenizer(tokenizer_name)
    renderer = get_renderer(
        renderer_name,
        tokenizer,
        image_processor=get_image_processor(args.model_name),
    )
    prompt_messages = (
        azure_openai_jsonl.prompt_messages(row)
        if args.data_path
        else [user_message(row)]
    )
    prompt = renderer.build_generation_prompt(prompt_messages)

    credential, credential_kwargs = _credential()
    client_kwargs = {
        "endpoint": args.project_endpoint,
        "credential": credential,
        **credential_kwargs,
    }
    if args.project_endpoint.startswith("http://"):
        client_kwargs["allow_insecure_http"] = True
    client = FineTuningSessionClient(**client_kwargs)
    session = FineTuningSession.create(
        client,
        base_model=args.model_name,
        lora_config=LoRAConfig(rank=args.lora_rank),
        type="training",
        training_type=args.training_type,
    )
    try:
        sampler_result = session.save_weights_for_sampler(seq_id=0, path="baseline")
        result = session.sample(
            prompt,
            SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                stop_criteria=renderer.get_stop_sequences(),
                seed=args.seed,
            ),
            checkpoint_id=sampler_result.checkpoint_id,
        )
        sequence = result.sequences[0]
        message, parsed = renderer.parse_response(sequence.tokens)
        answer = get_text_content(message)
        expected = str(row["answer"]).strip()
        print(f"question: {str(row['question']).strip()}")
        print(f"expected: {expected}")
        print(f"response: {answer}")
        print(f"parsed: {parsed}")
        exact_match = (
            azure_openai_jsonl.answers_match(answer, expected)
            if args.data_path
            else answers_match(answer, expected)
        )
        print(f"exact_match: {exact_match}")
    finally:
        session.close()
        client.close()


if __name__ == "__main__":
    main()