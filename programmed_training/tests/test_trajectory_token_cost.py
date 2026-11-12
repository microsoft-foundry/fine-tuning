from copy import deepcopy
from io import BytesIO

import pytest
from azure.ai.finetuningsessions.models import ImageChunk, ModelInput, ModelInputChunk
from PIL import Image

from programmed_training.completers import TokensWithLogprobs
from programmed_training.rl.data_processing import trajectory_token_cost_input
from programmed_training.rl.types import Trajectory, Transition


def transition(observation, action, *, force_new=False):
    return Transition(
        ob=ModelInput(chunks=[ModelInputChunk(tokens=observation)]),
        ac=TokensWithLogprobs(tokens=action, maybe_logprobs=[-0.1] * len(action)),
        reward=0.0, episode_done=False, force_new_training_sequence=force_new,
    )


def trajectory(transitions):
    return Trajectory(
        transitions=transitions,
        final_ob=ModelInput(chunks=[ModelInputChunk(tokens=list(range(1000)))]),
    )


def test_counts_sampled_history_once_and_excludes_unconsumed_final_observation():
    value = trajectory([
        transition([1, 2], [3, 4]),
        transition([1, 2, 3, 4, 5], [6]),
    ])
    before = deepcopy(value)

    assert trajectory_token_cost_input(value) == {
        "total_tokens": 6, "on_policy_tokens": 3, "prompt_tokens": 2,
    }
    assert value == before


@pytest.mark.parametrize(
    ("second", "expected_total"),
    [
        (transition([8, 9], [6]), 7),
        (transition([1, 2, 3, 4, 5], [6], force_new=True), 10),
    ],
)
def test_sequence_boundaries_count_newly_consumed_context(second, expected_total):
    assert trajectory_token_cost_input(trajectory([transition([1, 2], [3, 4]), second])) == {
        "total_tokens": expected_total, "on_policy_tokens": 3, "prompt_tokens": 2,
    }


def test_image_context_uses_expected_positions_not_encoded_bytes():
    value = transition([1, 2], [3, 4])
    data = BytesIO()
    Image.new("RGB", (2, 2)).save(data, format="PNG")
    value.ob.chunks.append(ImageChunk(data=data.getvalue(), format="png", expected_tokens=65))
    assert trajectory_token_cost_input(trajectory([value])) == {
        "total_tokens": 69, "on_policy_tokens": 2, "prompt_tokens": 67,
    }


def test_empty_trajectory_has_no_token_cost_input():
    assert trajectory_token_cost_input(trajectory([])) == {
        "total_tokens": 0, "on_policy_tokens": 0, "prompt_tokens": 0,
    }
