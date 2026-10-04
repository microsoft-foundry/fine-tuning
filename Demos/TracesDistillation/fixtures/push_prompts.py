"""Push N realistic retail-style prompts through a deployed Foundry agent
to generate App Insights traces that the Data Generation API can later
distill into SFT training data.

Auth: uses DefaultAzureCredential with Entra ID (for example, `az login`).

Usage:
  set AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
  python push_prompts.py --agent-name <name> --agent-version <v> --num-prompts 500
"""
import argparse, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT_TEMPLATES = [
    # Returns
    "I want to return my order {oid} — the {item} doesn't fit",
    "My order {oid} arrived damaged. The {item} has a cracked screen. Can I get a refund?",
    "I need to return {item} from order {oid} — wrong color shipped",
    "Order {oid}: the {item} stopped working after a week. Can I exchange?",
    "I changed my mind about {item} in order {oid} — what's my return window?",
    # Exchanges
    "Exchange {item} in order {oid} for a different size — I need a medium not a large",
    "Can I swap the {item} from order {oid} for the next model up? Willing to pay difference.",
    "I want to exchange the {item} from order {oid} — wrong product was shipped",
    # Cancellations
    "Cancel my order {oid}, I haven't received it yet and don't need it anymore",
    "I need to cancel order {oid} — placed by mistake",
    # Shipping disputes
    "My order {oid} says delivered but I never got it. {item} is missing.",
    "Tracking on order {oid} hasn't updated in a week. Where is my {item}?",
    "Order {oid} arrived late and I missed the event I needed the {item} for — can I get a credit?",
    # Multi-item
    "I want to return only one item from order {oid} — the {item} but keep the rest",
    "Two items from order {oid} arrived damaged: the {item} and the box of accessories",
    # Ambiguous
    "Help with order {oid}",
    "There's an issue with my recent purchase {oid}",
    "I'm not happy with order {oid}",
    # Policy questions
    "What's the return window for electronics on order {oid}?",
    "Am I eligible for a refund on {item} from order {oid}? I'm a Gold customer",
]

ITEMS = [
    "wireless headphones", "running shoes", "kitchen blender", "office chair", "smart watch",
    "wool sweater", "yoga mat", "espresso machine", "cordless drill", "hiking backpack",
    "winter jacket", "tablet stand", "noise-canceling earbuds", "memory foam pillow",
    "stainless steel water bottle", "laptop sleeve", "bluetooth speaker", "throw blanket",
]


def make_oid(rng):
    return f"ZA-{rng.randint(1000, 9999)}"


def push_one(responses_client, agent_name, agent_version, prompt, timeout=120):
    try:
        resp = responses_client.create(
            model=f"{agent_name}:{agent_version}" if agent_version else agent_name,
            input=prompt,
            store=False,
            timeout=timeout,
        )
        return True, getattr(resp, "id", "?"), None
    except Exception as e:
        return False, None, str(e)[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-name", required=True)
    p.add_argument("--agent-version", required=True)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--project-endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.project_endpoint:
        print("ERROR: AZURE_AI_PROJECT_ENDPOINT or --project-endpoint required", file=sys.stderr)
        return 2

    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    project = AIProjectClient(
        endpoint=args.project_endpoint,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )
    agent = project.agents.get_version(
        agent_name=args.agent_name,
        agent_version=args.agent_version,
    )
    responses = project.get_openai_client(agent_name=agent.name).responses

    rng = random.Random(args.seed)
    prompts = []
    for _ in range(args.num_prompts):
        tpl = rng.choice(PROMPT_TEMPLATES)
        prompts.append(tpl.format(oid=make_oid(rng), item=rng.choice(ITEMS)))

    print(f"Pushing {len(prompts)} prompts through {args.agent_name}{':'+args.agent_version if args.agent_version else ''}")
    print(f"  Project: {args.project_endpoint}")
    print(f"  Concurrency: {args.concurrency}")

    ok = 0
    fail = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(push_one, responses, args.agent_name, args.agent_version, pr): i
                   for i, pr in enumerate(prompts)}
        for done_i, fut in enumerate(as_completed(futures), 1):
            success, rid, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  err [{done_i}/{len(prompts)}]: {err}")
            if done_i % 25 == 0:
                elapsed = time.time() - start
                rate = done_i / elapsed if elapsed > 0 else 0
                eta = (len(prompts) - done_i) / rate if rate > 0 else 0
                print(f"  {done_i}/{len(prompts)}  ok={ok} fail={fail}  {rate:.1f}/s  ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s — ok={ok}  fail={fail}")
    print(f"\nNote: traces take ~90s to land in App Insights. Then run the datagen pipeline.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
