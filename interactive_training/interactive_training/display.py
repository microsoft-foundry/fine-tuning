import io

from azure.ai.finetuningsessions.models import Datum, ModelInputChunk
from termcolor import colored

from interactive_training.rl.types import Trajectory
from interactive_training.tokenizer_utils import Tokenizer
from interactive_training.utils.format_colorized import format_colorized


def to_ints(chunk: ModelInputChunk, tokenizer: Tokenizer):
    if isinstance(chunk, ModelInputChunk):
        return chunk.tokens
    else:
        (at_token,) = tokenizer.encode("@", add_special_tokens=False)
        return [at_token] * chunk.length


def colorize_example(datum: Datum, tokenizer: Tokenizer, key: str = "weights"):
    int_tokens = [
        token for chunk in datum.model_input.chunks for token in to_ints(chunk, tokenizer)
    ] + [int(datum.loss_fn_inputs["target_tokens"].data[-1])]
    weights = [0.0] + list(datum.loss_fn_inputs[key].data)
    return format_colorized(int_tokens, weights, tokenizer)


def format_trajectory(trajectory: Trajectory, tokenizer: Tokenizer) -> str:
    buf = io.StringIO()

    def colorize(s: str):
        return colored(s, "green", attrs=["bold"])

    def bprint(s: str):
        print(s, file=buf)

    bprint("=" * 60)
    for i, transition in enumerate(trajectory.transitions):
        bprint(f"------ Transition {i} ------")
        bprint(f"{colorize('Observation:')}: {tokenizer.decode([t for c in transition.ob.chunks for t in c.tokens])}")
        bprint(f"{colorize('Action:')}: {tokenizer.decode(transition.ac.tokens)}")
        bprint(f"{colorize('Reward:')}: {transition.reward}")
        bprint(f"{colorize('Episode done:')}: {transition.episode_done}")
        bprint(f"{colorize('Metrics:')}: {transition.metrics}")
        bprint("-" * 60)
    bprint("=" * 60)
    return buf.getvalue()
