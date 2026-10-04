import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential


PROMPTS = [
    "Order {order_id} arrived damaged. The headphones have a cracked case. Please refund them.",
    "The shoes in order {order_id} are the wrong size. Exchange them for size medium.",
    "Order {order_id} says delivered, but the Bluetooth speaker is missing. I need a replacement.",
    "Please cancel order {order_id}; it is still processing.",
    "The smartwatch in order {order_id} stopped working after one week. Please replace it.",
    "Order {order_id} was delivered five days late. Return the lamp and include any shipping credit.",
]


def tool_result(name, arguments):
    order_id = arguments.get("order_id", "ZA-0000")
    if name == "get_order_details":
        return {
            "order_id": order_id,
            "customer": {"loyalty_tier": "Gold"},
            "items": [
                {
                    "item_id": "ITEM-1",
                    "sku": "SKU-DEMO-1",
                    "name": "demo item",
                    "category": "electronics",
                    "price": 129.0,
                    "final_sale": False,
                }
            ],
            "payment_method": "card",
        }
    if name == "get_fulfillment_status":
        return {
            "order_id": order_id,
            "status": "delivered",
            "late_delivery": True,
            "days_late": 5,
            "days_since_delivery": 7,
        }
    if name == "check_resolution_policy":
        return {
            "order_id": order_id,
            "item_id": arguments.get("item_id", "ITEM-1"),
            "eligible": True,
            "allowed_actions": ["refund", "exchange", "replacement"],
            "restocking_fee_percent": 0,
        }
    if name == "check_inventory":
        return {"sku": arguments.get("sku", "SKU-DEMO-1"), "available": True, "quantity": 25}
    if name == "calculate_resolution":
        return {
            "order_id": order_id,
            "approved": True,
            "refund_amount": 129.0,
            "shipping_credit": 10.0,
            "restocking_fee": 0.0,
        }
    if name == "submit_resolution":
        return {
            "order_id": order_id,
            "status": "submitted",
            "resolution_id": f"RES-{order_id}",
        }
    return {"ok": True}


def run_conversation(client, model, prompt, max_turns=8):
    response = client.responses.create(model=model, input=prompt, store=True)
    response_ids = [response.id]
    tool_calls = 0
    for _ in range(max_turns):
        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not calls:
            return {
                "status": response.status,
                "response_ids": response_ids,
                "tool_calls": tool_calls,
                "output_text": response.output_text,
            }
        outputs = []
        for call in calls:
            arguments = json.loads(call.arguments or "{}")
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(tool_result(call.name, arguments)),
                }
            )
            tool_calls += 1
        response = client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=outputs,
            store=True,
        )
        response_ids.append(response.id)
    raise RuntimeError(f"Conversation exceeded {max_turns} tool rounds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-endpoint", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--conversations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="run/trace_generation.json")
    args = parser.parse_args()

    project = AIProjectClient(
        endpoint=args.project_endpoint,
        credential=AzureCliCredential(),
        allow_preview=True,
    )
    client = project.get_openai_client(agent_name=args.agent_name)
    rng = random.Random(42)
    prompts = [
        rng.choice(PROMPTS).format(order_id=f"ZA-{1000 + i}")
        for i in range(args.conversations)
    ]

    started = time.time()
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_conversation, client, args.model, prompt): (index, prompt)
            for index, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            index, prompt = futures[future]
            try:
                result = future.result()
                result.update({"index": index, "prompt": prompt})
                results.append(result)
                print(
                    f"[{len(results) + len(failures)}/{len(prompts)}] "
                    f"ok tool_calls={result['tool_calls']} responses={len(result['response_ids'])}"
                )
            except Exception as exc:
                failures.append({"index": index, "prompt": prompt, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{len(results) + len(failures)}/{len(prompts)}] failed: {failures[-1]['error']}")

    report = {
        "project_endpoint": args.project_endpoint,
        "agent_name": args.agent_name,
        "model": args.model,
        "requested_conversations": args.conversations,
        "successful_conversations": len(results),
        "failed_conversations": len(failures),
        "tool_calls": sum(item["tool_calls"] for item in results),
        "response_ids": [response_id for item in results for response_id in item["response_ids"]],
        "duration_seconds": round(time.time() - started, 2),
        "results": sorted(results, key=lambda item: item["index"]),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "response_ids")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
