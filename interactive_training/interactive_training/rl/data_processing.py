"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from typing import List

import torch
from azure.ai.finetuningsessions.models import Datum, ImageChunk, ModelInput, ModelInputChunk, TensorData
from interactive_training.rl.types import Trajectory, TrajectoryGroup
from interactive_training.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from interactive_training.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | ModelInputChunk | ImageChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        elif isinstance(elem, ImageChunk):
            out += elem.expected_tokens
        else:
            out += len(elem.tokens)
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> ModelInput:
    out: list[ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            # Must copy the list — ModelInputChunk stores a reference,
            # and .clear() below would empty the chunk's tokens.
            out.append(ModelInputChunk(tokens=list(current_text_chunk)))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return ModelInput(chunks=out)


def _flatten_chunks(chunks: list[ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, ModelInputChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(
    traj: Trajectory,
    traj_advantage: float,
    token_advantage_adjustments: list[list[float]] | None = None,
) -> list[Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            sum(
                len(chunk.tokens)
                if isinstance(chunk, ModelInputChunk)
                else chunk.expected_tokens
                for chunk in input_tokens_T.chunks
            )
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData(data=torch.tensor(target_tokens_T).detach().cpu().tolist()),
                "logprobs": TensorData(data=torch.tensor(sampled_logprobs_T).detach().cpu().tolist()),
                "advantages": TensorData(data=torch.tensor(advantages_T).detach().cpu().tolist()),
                "mask": TensorData(data=torch.tensor(mask_T).detach().cpu().tolist()),
            },
        )

    if token_advantage_adjustments is not None:
        if len(token_advantage_adjustments) != len(traj.transitions) or any(
            len(values) != len(transition.ac.tokens)
            for transition, values in zip(
                traj.transitions, token_advantage_adjustments, strict=True
            )
        ):
            raise ValueError(
                "token advantage adjustments must align with trajectory transitions and action tokens"
            )

    data: list[Datum] = []
    for transition_index, transition in enumerate(traj.transitions):
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif transition.force_new_training_sequence:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        action_advantages = [traj_advantage] * len(ac_with_logprobs.tokens)
        if token_advantage_adjustments is not None:
            action_advantages = [
                traj_advantage + token_adjustment
                for token_adjustment in token_advantage_adjustments[transition_index]
            ]
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + action_advantages
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
    token_advantage_adjustments_P: list[
        list[list[list[float]]] | None
    ]
    | None = None,
) -> tuple[List[Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[Datum] = []
    metadata_D: list[dict[str, int]] = []

    if token_advantage_adjustments_P is not None and len(
        token_advantage_adjustments_P
    ) != len(trajectory_groups_P):
        raise ValueError(
            "token advantage adjustment groups must align with trajectory groups"
        )
    group_token_advantage_adjustments = (
        [None] * len(trajectory_groups_P)
        if token_advantage_adjustments_P is None
        else token_advantage_adjustments_P
    )
    for i_group, (
        traj_group,
        advantages_G,
        token_advantage_adjustments_G,
    ) in enumerate(
        zip(
            trajectory_groups_P,
            advantages_P,
            group_token_advantage_adjustments,
            strict=True,
        )
    ):
        if token_advantage_adjustments_G is not None and len(
            token_advantage_adjustments_G
        ) != len(traj_group.trajectories_G):
            raise ValueError(
                "token advantage adjustments must align with surviving trajectories"
            )
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(
                traj,
                float(traj_advantage),
                token_advantage_adjustments=(
                    None
                    if token_advantage_adjustments_G is None
                    else token_advantage_adjustments_G[i_traj]
                ),
            )
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
